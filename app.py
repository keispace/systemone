#!/usr/bin/env python3
"""System One 계열 모델과 로컬 Laya를 비교하는 웹 PoC.

로컬 Laya Multilingual은 시작 시 MPS(기본값)에 preload하고, 외부 모델은
provider adapter를 통해 같은 typed question UI에서 호출한다.

예:
  .venv/bin/python app.py
  .venv/bin/python app.py --device cpu --port 8766
  .venv/bin/python app.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

from dotenv import dotenv_values


MAX_BODY_BYTES = 64 * 1024
MAX_TEXT_CHARS = 4_000
MAX_INSTRUCTIONS_CHARS = 1_000
MAX_CRITERION_CHARS = 300
MAX_OPTIONS = 12
REMOTE_TIMEOUT_SECONDS = 20
DECISIONS_MAX_BODY_BYTES = 32 * 1024
PROJECT_ROOT = Path(__file__).resolve().parent

# The UI exposes only these stable ids.  Provider wire ids and endpoint selection live
# on the server so a browser cannot redirect a request or select an unreviewed model.
MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "local-laya-multilingual": {
        "id": "local-laya-multilingual", "provider": "local", "wire_model": "multilingual",
        "location": "이 Mac / local", "name": "Laya Multilingual",
        "select_label": "(local) Laya Multilingual",
        "detail": "multilingual checkpoint · lang=ko", "question_types": ("choice", "noul", "score"),
    },
    "decisions-typesafe-jev-1-13": {
        "id": "decisions-typesafe-jev-1-13", "provider": "decisions", "wire_model": "typesafe/jev-1.13",
        "location": "Decisions API / remote", "name": "Jev 1.13",
        "select_label": "(decisions-api.org) Jev (1.13)",
        "detail": "모델 주체 TypeSafe · wire ID typesafe/jev-1.13", "question_types": ("choice", "noul", "score"),
    },
    "typesafe-jev-latest": {
        "id": "typesafe-jev-latest", "provider": "typesafe", "wire_model": "jev-latest",
        "location": "TypeSafe API / remote", "name": "Jev",
        "select_label": "(typesafe.ai) Jev (latest)",
        "detail": "TypeSafe 직접 서빙 · wire ID jev-latest", "question_types": ("choice", "noul", "score"),
    },
}


class InputError(ValueError):
    """A request the user can correct; its text is safe to return as a 400 response."""


class UpstreamError(RuntimeError):
    """A safe description of an upstream failure; never include upstream body or keys."""


def wire_json_bytes(body: Dict[str, Any]) -> bytes:
    """The one JSON encoding used for Decisions size validation and HTTP transmission."""
    return json.dumps(body, ensure_ascii=False).encode("utf-8")


def load_project_dotenv(environ: Dict[str, str] | None = None, dotenv_path: Path | None = None) -> None:
    """Load only approved remote API keys without replacing exported values."""
    target = os.environ if environ is None else environ
    values = dotenv_values(dotenv_path or PROJECT_ROOT / ".env")
    for key_name in ("DECISION_ORG_API_KEY", "TYPESAFE_API_KEY"):
        value = values.get(key_name)
        # Presence (including an intentionally empty string) has override=False semantics.
        if value is not None and key_name not in target:
            target[key_name] = value


def _string(payload: Dict[str, Any], key: str, label: str, maximum: int, *, required: bool = True) -> str:
    value = payload.get(key, "")
    if value is None and not required:
        return ""
    if not isinstance(value, str):
        raise InputError("%s은(는) 문자열이어야 합니다." % label)
    value = value.strip()
    if required and not value:
        raise InputError("%s을(를) 입력해 주세요." % label)
    if len(value) > maximum:
        raise InputError("%s은(는) %d자 이하여야 합니다." % (label, maximum))
    return value


def parse_line_criteria(raw: Any, label: str, *, require_labels: bool) -> List[Tuple[str, str]]:
    """Turn editable `label: description` lines into an ordered list.

    The labels must be unique because Laya returns choice probabilities keyed by them.
    A colon is optional; then the same text is both label and explanation.
    """
    if not isinstance(raw, str):
        raise InputError("%s은(는) 줄마다 텍스트로 입력해 주세요." % label)
    result: List[Tuple[str, str]] = []
    seen = set()
    for index, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        name, sep, description = line.partition(":")
        name = name.strip()
        description = description.strip() if sep else name
        if not name:
            raise InputError("%s %d번째 줄의 라벨이 비어 있습니다." % (label, index))
        if len(name) > MAX_CRITERION_CHARS or len(description) > MAX_CRITERION_CHARS:
            raise InputError("%s의 라벨과 설명은 각각 %d자 이하여야 합니다." % (label, MAX_CRITERION_CHARS))
        # Choice returns labels as JSON keys; score has no label key, so its rendered
        # description is the meaningful option identity. Reject duplicates in both forms.
        identity = name if require_labels else description
        if identity in seen:
            raise InputError("%s에 중복된 선택지가 있습니다: %s" % (label, identity))
        seen.add(identity)
        result.append((name, description))
    if len(result) < 2:
        raise InputError("%s은(는) 최소 두 줄이 필요합니다." % label)
    if len(result) > MAX_OPTIONS:
        raise InputError("%s은(는) 최대 %d개까지 입력할 수 있습니다." % (label, MAX_OPTIONS))
    return result


def parse_threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise InputError("참/거짓 threshold는 0과 1 사이의 숫자여야 합니다.")
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        raise InputError("참/거짓 threshold는 0과 1 사이의 숫자여야 합니다.") from None
    if not 0.0 <= threshold <= 1.0:
        raise InputError("참/거짓 threshold는 0 이상 1 이하여야 합니다.")
    return threshold


def build_payload(payload: Any) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Validate browser JSON and build exactly the state/questions Laya consumes.

    This has no Laya import so --self-test can exercise the important boundary without
    downloading or constructing a checkpoint.
    """
    if not isinstance(payload, dict):
        raise InputError("요청 본문은 JSON 객체여야 합니다.")
    text = _string(payload, "text", "입력", MAX_TEXT_CHARS)
    instructions = _string(payload, "instructions", "판정 질문", MAX_INSTRUCTIONS_CHARS)
    qtype = payload.get("type")
    if qtype not in ("choice", "noul", "score"):
        raise InputError("판정 형식은 선택지, 참/거짓, 점수 중 하나여야 합니다.")

    question: Dict[str, Any] = {"type": qtype, "instructions": instructions}
    presentation: Dict[str, Any] = {"type": qtype}
    if qtype == "choice":
        options = parse_line_criteria(payload.get("criteria", ""), "선택지", require_labels=True)
        question["criteria"] = dict(options)
        presentation["criteria"] = dict(options)
    elif qtype == "score":
        levels = parse_line_criteria(payload.get("criteria", ""), "점수 등급", require_labels=False)
        # Score order remains 0, 1, ... in Laya; retain the human label in each legend item.
        question["criteria"] = [name if name == description else "%s: %s" % (name, description)
                                for name, description in levels]
        presentation["criteria"] = question["criteria"]
    else:
        false_criterion = _string(payload, "false_criterion", "거짓 기준", MAX_CRITERION_CHARS)
        true_criterion = _string(payload, "true_criterion", "참 기준", MAX_CRITERION_CHARS)
        question["criteria"] = {"false": false_criterion, "true": true_criterion}
        presentation["criteria"] = question["criteria"]
        presentation["threshold"] = parse_threshold(payload.get("threshold", 0.5))
    # This UI deliberately has one input and one question. The keys are implementation
    # details, not knobs a user needs to configure, but are still visible in raw payload.
    return {"input": text}, {"judgement": question}, presentation


def resolve_model(model_id: Any) -> Dict[str, Any]:
    if not isinstance(model_id, str) or model_id not in MODEL_CATALOG:
        raise InputError("알 수 없는 모델입니다. 목록에서 다시 선택해 주세요.")
    return MODEL_CATALOG[model_id]


def check_capability(model: Dict[str, Any], questions: Dict[str, Dict[str, Any]]) -> None:
    for question in questions.values():
        if question["type"] not in model["question_types"]:
            raise InputError("선택한 모델은 이 판정 형식을 지원하지 않습니다.")
        if question["type"] == "score" and len(question["criteria"]) > 10 and model["provider"] in ("decisions", "typesafe"):
            raise InputError("선택한 Jev 원격 모델의 점수 등급은 최대 10개입니다.")


def normalize_result(model: Dict[str, Any], raw: Dict[str, Any], latency_ms: float,
                     provider_metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if not isinstance(raw, dict) or not isinstance(raw.get("model"), str) or not isinstance(raw.get("answers"), dict):
        raise UpstreamError("원격 모델 응답 형식이 올바르지 않습니다.")
    return {
        "requested_catalog_id": model["id"], "provider": model["provider"],
        "requested_wire_model": model["wire_model"], "resolved_model": raw["model"],
        "answers": raw["answers"], "usage": raw.get("usage"),
        "provider_metadata": provider_metadata or {}, "latency_ms": round(latency_ms, 2),
    }


def valid_probabilities(probabilities: Any) -> bool:
    return isinstance(probabilities, dict) and bool(probabilities) and all(
        not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1
        for value in probabilities.values())


def public_result(normalized: Dict[str, Any], question_id: str, presentation: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a provider-neutral System One result to the existing UI shape."""
    answer = normalized["answers"].get(question_id)
    if not isinstance(answer, dict) or answer.get("type") not in ("choice", "noul", "score"):
        raise UpstreamError("원격 모델 응답에 판정 답변이 없습니다.")
    result: Dict[str, Any] = {
        "type": answer["type"],
        "sdk_confidence": answer.get("confidence"), "latency_ms": normalized["latency_ms"],
        "model": {key: normalized[key] for key in ("requested_catalog_id", "provider", "requested_wire_model", "resolved_model")},
        "provider_metadata": normalized["provider_metadata"], "usage": normalized["usage"],
        "raw_payload": {"model": normalized["requested_wire_model"], "state": presentation["state"], "questions": presentation["questions"]},
        "raw_result": {"model": normalized["resolved_model"], "answers": normalized["answers"], "usage": normalized["usage"]},
    }
    if answer["type"] == "choice":
        probabilities = answer.get("probabilities")
        choice = answer.get("choice")
        if not valid_probabilities(probabilities) or not isinstance(choice, str) or choice not in probabilities:
            raise UpstreamError("선택지 답변 형식이 올바르지 않습니다.")
        result.update({
            "selected_label": answer["choice"],
            "probabilities": probabilities,
            "maximum_probability": max(probabilities.values()),
        })
    elif answer["type"] == "noul":
        true_probability = answer.get("noul")
        if isinstance(true_probability, bool) or not isinstance(true_probability, (int, float)) or not 0 <= true_probability <= 1:
            raise UpstreamError("참/거짓 답변 형식이 올바르지 않습니다.")
        threshold = presentation["threshold"]
        result.update({
            "true_probability": true_probability,
            "false_probability": round(1.0 - true_probability, 4),
            "threshold": threshold,
            "boolean": true_probability >= threshold,
        })
    else:
        probabilities, legend = answer.get("probabilities"), answer.get("legend")
        score = answer.get("score")
        if (not valid_probabilities(probabilities) or not isinstance(legend, dict) or not legend
                or set(legend) != set(probabilities) or isinstance(score, bool)
                or not isinstance(score, (int, float)) or not math.isfinite(score)
                or not 0 <= score <= len(legend) - 1):
            raise UpstreamError("점수 답변 형식이 올바르지 않습니다.")
        result.update({"expected_score": answer["score"], "legend": legend,
                       "probabilities": probabilities, "maximum_probability": max(probabilities.values())})
    return result


HTML = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>System One 모델 PoC</title>
<style>
:root{--ink:#192229;--muted:#66727a;--paper:#f7f5ef;--card:#fffdf8;--line:#d8d4c9;--accent:#0d6b64;--bad:#a8392f;--good:#0d7048}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Apple SD Gothic Neo",sans-serif}.wrap{max-width:1040px;margin:0 auto;padding:32px 20px 72px}h1{margin:0;font-size:28px;letter-spacing:-.04em}h2{font-size:17px;margin:0 0 12px}.sub,.hint{color:var(--muted)}.notice{border-left:4px solid #db9f25;background:#fff7df;padding:12px 14px;margin:20px 0;border-radius:4px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;box-shadow:0 2px 8px #352d1710}.wide{grid-column:1/-1}label{display:block;font-weight:700;margin:0 0 5px}input,textarea,select,button{font:inherit}input,textarea,select{width:100%;border:1px solid #b9b5aa;border-radius:6px;background:#fff;padding:9px 10px;color:inherit}textarea{min-height:112px;resize:vertical}.small{min-height:90px}.field{margin-bottom:14px}.help{font-size:12px;color:var(--muted);margin:5px 0 0}button{border:0;border-radius:7px;background:var(--accent);color:white;padding:10px 16px;font-weight:700;cursor:pointer}button:disabled{opacity:.55;cursor:wait}.presets{display:flex;flex-wrap:wrap;gap:8px}.preset{background:#e4f0ed;color:#13554f}.status{display:inline-block;margin-left:10px;color:var(--muted)}.hidden{display:none}.error{color:var(--bad);background:#ffefeb;border:1px solid #efc2ba;border-radius:6px;padding:10px;margin:12px 0}.result{margin-top:22px}.answer{font-size:25px;font-weight:800;letter-spacing:-.03em}.badge{display:inline-block;border-radius:999px;padding:3px 10px;font-weight:800;font-size:13px}.true{background:#dff5e6;color:var(--good)}.false{background:#fee7e3;color:var(--bad)}.metric{display:inline-block;margin:9px 18px 0 0}.metric b{display:block;font-size:17px}.barrow{display:grid;grid-template-columns:minmax(70px,160px) 1fr 56px;gap:8px;align-items:center;margin:7px 0}.bar{height:11px;background:#e8e5dc;border-radius:999px;overflow:hidden}.bar i{display:block;height:100%;background:#16877d;border-radius:999px}details{margin-top:13px;background:#f2f0e9;border-radius:6px;padding:8px 10px}summary{cursor:pointer;font-weight:700}pre{white-space:pre-wrap;word-break:break-word;font-size:12px;margin:10px 0 2px}@media(max-width:700px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}.wrap{padding:22px 13px}}
</style></head><body><main class="wrap">
<h1>System One 판정 실험</h1><p class="sub">브라우저는 이 Mac의 <code>127.0.0.1</code> 서버에만 연결됩니다. 모델·키·원격 API 주소는 서버가 관리합니다.</p>
<div class="notice"><b>해석 주의.</b> 임의의 질문·기준에 대한 모델 추정치입니다. 업무 라우팅, 환불 여부, 긴급도를 포함한 모든 결과는 실제 도메인의 라벨 표본으로 정확도를 평가하기 전에는 데모 결과입니다.</div>
<form id="form" class="grid">
<section class="card wide"><h2>모델</h2><div class="field"><label for="model">실행 모델</label><select id="model"><option value="local-laya-multilingual">(local) Laya Multilingual</option></select><p id="model_metadata" class="help">모델 목록을 불러오는 중…</p></div><div id="remote_warning" class="notice hidden" role="status"></div></section>
<section class="card"><h2>입력</h2><div class="field"><label for="text">입력</label><textarea id="text" maxlength="4000">결제가 두 번 됐습니다. 중복 결제분을 취소하고 환불해 주세요.</textarea></div></section>
<section class="card"><h2>판정 질문</h2><div class="field"><label for="instructions">판정 질문</label><textarea id="instructions" class="small" maxlength="1000">이 문의를 담당할 부서는?</textarea></div></section>
<section class="card wide"><h2>판정 형식과 기준</h2><div class="field"><label>업무 예제 프리셋</label><div class="presets"><button id="preset_choice" class="preset" type="button">선택지 예제</button><button id="preset_noul" class="preset" type="button">참·거짓 예제</button><button id="preset_score" class="preset" type="button">점수 예제</button></div><p class="help">버튼을 누를 때만 입력·질문·기준을 해당 업무 예제로 채웁니다.</p></div><div class="field"><label for="type">형식</label><select id="type"><option value="choice">선택지 (choice)</option><option value="noul">참/거짓 (noul)</option><option value="score">점수 (score)</option></select><p class="help">세 형식 모두 후보별 확률 분포를 만듭니다. 선택지는 최고 확률 라벨, 참/거짓은 두 후보와 UI threshold, 점수는 순서 있는 후보 분포의 기대값으로 요약합니다.</p></div>
<div id="choice_fields" class="field"><label for="criteria">선택지</label><textarea id="criteria" class="small">결제: 결제, 중복 청구, 취소 또는 환불 요청
계정: 로그인, 계정 접근, 비밀번호 또는 인증 문제
기술지원: 오류, 장애, 기능 동작 또는 서비스 연동 문제
개인정보: 개인정보 열람, 정정, 삭제 또는 보호 요청</textarea><p class="help">한 줄에 <code>라벨: 짧은 기준</code>을 씁니다. 콜론이 없으면 그 줄 전체를 라벨과 기준으로 사용합니다.</p></div>
<div id="noul_fields" class="hidden"><div class="field"><label for="false_criterion">거짓 기준</label><input id="false_criterion" value="환불 요청이 없고 정보 문의나 다른 문제인 경우"></div><div class="field"><label for="true_criterion">참 기준</label><input id="true_criterion" value="결제 취소 또는 환불을 명시적으로 요청한 경우"></div><div class="field"><label for="threshold">T/F threshold</label><input id="threshold" type="number" min="0" max="1" step="0.01" value="0.5"><p class="help">기본 0.5는 이 UI가 표시용으로 적용하는 규칙이며 Laya 모델 계약 자체가 아닙니다.</p></div></div>
<div id="score_fields" class="hidden"><div class="field"><label for="score_criteria">점수 등급 (낮음 → 높음)</label><textarea id="score_criteria" class="small">낮음: 불편이 작고 즉시 처리할 필요가 없음
보통: 일부 기능 영향 또는 일반 문의
높음: 중요한 기능 장애 또는 업무에 큰 영향
긴급: 전체 장애 또는 모든 직원의 업무 중단</textarea><p class="help">줄의 순서가 0, 1, 2… 점수입니다. 각 줄의 등급명과 기준이 Laya legend에 순서대로 전달됩니다.</p></div></div>
<button id="submit" type="submit">한국어로 판정 실행</button><span id="status" class="status">서버 상태 확인 중…</span><div id="error" class="error hidden" role="alert"></div></section></form>
<section id="result" class="card result hidden" aria-live="polite"></section></main>
<script>
const $=id=>document.getElementById(id), type=$('type'), model=$('model'), status=$('status'), error=$('error'), result=$('result'), submit=$('submit');let catalog=[];
const PRESETS={choice:{type:'choice',text:'결제가 두 번 됐습니다. 중복 결제분을 취소하고 환불해 주세요.',instructions:'이 문의를 담당할 부서는?',criteria:'결제: 결제, 중복 청구, 취소 또는 환불 요청\n계정: 로그인, 계정 접근, 비밀번호 또는 인증 문제\n기술지원: 오류, 장애, 기능 동작 또는 서비스 연동 문제\n개인정보: 개인정보 열람, 정정, 삭제 또는 보호 요청'},noul:{type:'noul',text:'지난달 결제가 두 번 청구됐습니다. 중복 결제분을 환불해 주세요.',instructions:'이 입력은 환불을 요청하는가?',false_criterion:'환불 요청이 없고 정보 문의나 다른 문제인 경우',true_criterion:'결제 취소 또는 환불을 명시적으로 요청한 경우',threshold:'0.5'},score:{type:'score',text:'전체 서비스가 접속되지 않아 모든 직원의 업무가 중단되었습니다.',instructions:'이 문의의 긴급도는?',criteria:'낮음: 불편이 작고 즉시 처리할 필요가 없음\n보통: 일부 기능 영향 또는 일반 문의\n높음: 중요한 기능 장애 또는 업무에 큰 영향\n긴급: 전체 장애 또는 모든 직원의 업무 중단'}};
function esc(x){return String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function pct(x){return (Number(x)*100).toFixed(2)+'%';}
function bars(items){return Object.entries(items).map(([k,v])=>`<div class="barrow"><span>${esc(k)}</span><div class="bar"><i style="width:${Math.max(0,Math.min(100,Number(v)*100))}%"></i></div><b>${pct(v)}</b></div>`).join('');}
function clearResult(){result.textContent='';result.classList.add('hidden');}
function selectedModel(){return catalog.find(x=>x.id===model.value);}
function modelDetail(){const m=selectedModel();if(!m)return;$('model_metadata').textContent=`위치: ${m.location} / 모델명: ${m.name} / 상세명: ${m.detail} / 지원: ${m.question_types.join(', ')}${m.available?'':' / '+(m.reason||'사용 불가')}`;const remote=m.provider!=='local';$('remote_warning').classList.toggle('hidden',!remote);if(remote){const provider=m.provider==='decisions'?'Decisions API':'TypeSafe API';$('remote_warning').textContent=`주의: ${provider} 원격 모델은 크레딧을 사용할 수 있으며, 제출한 입력과 판정 기준이 ${provider}로 전송됩니다.`;}}
function syncModels(){const v=type.value;for(const option of model.options){const m=catalog.find(x=>x.id===option.value);option.disabled=!m||!m.available||!m.question_types.includes(v);}if(model.selectedOptions[0]?.disabled){const next=[...model.options].find(o=>!o.disabled);if(next)model.value=next.value;}modelDetail();}
function switchType(){const v=type.value;$('choice_fields').classList.toggle('hidden',v!=='choice');$('noul_fields').classList.toggle('hidden',v!=='noul');$('score_fields').classList.toggle('hidden',v!=='score');syncModels();clearResult();}
function applyPreset(name){const p=PRESETS[name];type.value=p.type;$('text').value=p.text;$('instructions').value=p.instructions;if(p.type==='choice')$('criteria').value=p.criteria;if(p.type==='score')$('score_criteria').value=p.criteria;if(p.type==='noul'){$('false_criterion').value=p.false_criterion;$('true_criterion').value=p.true_criterion;$('threshold').value=p.threshold;}switchType();}
type.addEventListener('change',switchType);model.addEventListener('change',()=>{modelDetail();clearResult();});$('form').addEventListener('input',clearResult);$('form').addEventListener('change',clearResult);$('preset_choice').addEventListener('click',()=>applyPreset('choice'));$('preset_noul').addEventListener('click',()=>applyPreset('noul'));$('preset_score').addEventListener('click',()=>applyPreset('score'));
async function health(){try{const r=await fetch('/api/health');const d=await r.json();catalog=d.models||[];model.replaceChildren(...catalog.map(m=>{const o=document.createElement('option');o.value=m.id;o.textContent=m.select_label;return o;}));model.value='local-laya-multilingual';switchType();status.textContent=d.ready?'준비됨: '+d.loaded.join(', '):'로컬 모델 준비 중';}catch(e){status.textContent='서버에 연결할 수 없습니다.';}}
function payload(){const v=type.value;const p={model_id:model.value,text:$('text').value,instructions:$('instructions').value,type:v};if(v==='choice')p.criteria=$('criteria').value;if(v==='score')p.criteria=$('score_criteria').value;if(v==='noul'){p.false_criterion=$('false_criterion').value;p.true_criterion=$('true_criterion').value;p.threshold=$('threshold').value;}return p;}
function details(title,obj){return `<details><summary>${title}</summary><pre>${esc(JSON.stringify(obj,null,2))}</pre></details>`;}
function confidence(d){return d.sdk_confidence==null?'—':pct(d.sdk_confidence);}
function draw(d){let top='';if(d.type==='choice')top=`<div class="answer">${esc(d.selected_label)}</div><div class="metric">최고 선택지 확률<b>${pct(d.maximum_probability)}</b></div><div class="metric">SDK confidence<b>${confidence(d)}</b></div><h2>라벨별 확률</h2>${bars(d.probabilities)}`;else if(d.type==='noul'){top=`<div class="answer"><span class="badge ${d.boolean?'true':'false'}">${d.boolean?'T (참)':'F (거짓)'}</span></div><div class="metric">p(true)<b>${pct(d.true_probability)}</b></div><div class="metric">p(false)<b>${pct(d.false_probability)}</b></div><div class="metric">UI threshold<b>${pct(d.threshold)}</b></div><div class="metric">SDK confidence<b>${confidence(d)}</b></div>`;}else {const legend=Object.fromEntries(Object.entries(d.legend).map(([k,v])=>['level '+k+': '+v,d.probabilities[k]]));top=`<div class="answer">기대 점수 ${Number(d.expected_score).toFixed(2)} / ${Object.keys(d.legend).length-1}</div><div class="metric">최고 등급 확률<b>${pct(d.maximum_probability)}</b></div><div class="metric">SDK confidence<b>${confidence(d)}</b></div><h2>등급별 확률</h2>${bars(legend)}`;}const m=d.model||{};result.innerHTML=`<h2>판정 결과</h2>${top}<p class="hint">선택 모델: ${esc(m.requested_catalog_id||'알 수 없음')} · 실행 모델: ${esc(m.resolved_model||'알 수 없음')} · 제공자: ${esc(m.provider||'')} · 예측 지연 ${esc(d.latency_ms)} ms · 입력 토큰 ${esc((d.usage||{}).input_tokens??'?')}</p>${details('원시 payload',d.raw_payload)}${details('원시 결과',d.raw_result)}${details('제공자 메타데이터',d.provider_metadata)}`;result.classList.remove('hidden');}
$('form').addEventListener('submit',async e=>{e.preventDefault();error.classList.add('hidden');submit.disabled=true;const m=selectedModel();status.textContent=m?.provider==='local'?'MPS에서 예측 중…':`${m?.provider==='decisions'?'Decisions':'TypeSafe'} API에 요청 중…`;try{const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload())});const d=await r.json();if(!r.ok)throw new Error(d.error||'요청을 처리하지 못했습니다.');draw(d);status.textContent='준비됨';}catch(e){error.textContent=e.message;error.classList.remove('hidden');status.textContent='입력을 확인해 주세요.';}finally{submit.disabled=false;}});health();
</script></body></html>'''


class LayaService:
    """Owns the local Router and server-only remote provider adapters."""

    def __init__(self, device: str, *, transport: Any = None, environ: Dict[str, str] | None = None):
        self.device = device
        self.router = None
        self.inference_lock = threading.Lock()
        self.ready = False
        self.error = None
        self.transport = transport or self._http_transport
        self.environ = os.environ if environ is None else environ

    def start(self) -> None:
        # Delayed import keeps --help and --self-test free of model/runtime startup.
        from laya.router import Router

        self.router = Router(device=self.device, max_loaded=1)
        self.router.preload([MODEL_CATALOG["local-laya-multilingual"]["wire_model"]])
        self.ready = True

    @staticmethod
    def _http_transport(url: str, headers: Dict[str, str], body: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        encoded = wire_json_bytes(body)
        req = urlrequest.Request(url, data=encoded, headers=headers, method="POST")
        try:
            with urlrequest.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            raise UpstreamError("원격 모델이 HTTP %d 오류를 반환했습니다." % exc.code) from None
        except (urlerror.URLError, TimeoutError, json.JSONDecodeError):
            raise UpstreamError("원격 모델에 연결하거나 응답을 해석하지 못했습니다.") from None

    def _configured(self, model: Dict[str, Any]) -> bool:
        key_by_provider = {
            "decisions": "DECISION_ORG_API_KEY",
            "typesafe": "TYPESAFE_API_KEY",
        }
        return model["provider"] == "local" or bool(self.environ.get(key_by_provider.get(model["provider"], "")))

    def catalog_for_browser(self) -> List[Dict[str, Any]]:
        items = []
        for item in MODEL_CATALOG.values():
            configured = self._configured(item)
            available = self.ready if item["provider"] == "local" else configured
            items.append({key: item[key] for key in ("id", "provider", "location", "name", "detail", "select_label", "question_types")}
                         | {"configured": configured, "available": available,
                            "reason": None if available else ("로컬 모델 준비 중" if item["provider"] == "local" else "API 키 미설정")})
        return items

    def _local(self, model: Dict[str, Any], state: Dict[str, str], questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        if not self.ready or self.router is None:
            raise RuntimeError("Laya multilingual 모델이 아직 준비되지 않았습니다.")
        with self.inference_lock:
            started = time.perf_counter()
            raw = self.router.predict(state, questions, lang="ko")
            latency_ms = (time.perf_counter() - started) * 1000
        raw = dict(raw)
        routing = raw.get("routing") if isinstance(raw.get("routing"), dict) else None
        raw["model"] = routing.get("model", model["wire_model"]) if routing else model["wire_model"]
        return normalize_result(model, raw, latency_ms, {"routing": routing})

    def _decisions(self, model: Dict[str, Any], state: Dict[str, str], questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        key = self.environ.get("DECISION_ORG_API_KEY")
        if not key:
            raise InputError("선택한 원격 모델의 API 키가 설정되지 않았습니다.")
        body = {"model": model["wire_model"], "state": state, "questions": questions}
        if len(wire_json_bytes(body)) > DECISIONS_MAX_BODY_BYTES:
            raise InputError("Decisions Jev 요청 본문은 32 KiB 이하여야 합니다.")
        headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
        started = time.perf_counter()
        response = self.transport("https://decisions-api.org/v1/systemone", headers, body, REMOTE_TIMEOUT_SECONDS)
        latency_ms = (time.perf_counter() - started) * 1000
        if not isinstance(response, dict) or response.get("code") != 0:
            raise UpstreamError("Decisions API가 예측을 완료하지 못했습니다.")
        data = response.get("data")
        if not isinstance(data, dict) or not isinstance(data.get("result"), dict):
            raise UpstreamError("Decisions API 응답 형식이 올바르지 않습니다.")
        core = data["result"]
        metadata = {key: data[key] for key in ("requestId", "creditsUsed", "historySaved") if key in data}
        if "elapsedMs" in core:
            metadata["elapsedMs"] = core["elapsedMs"]
        return normalize_result(model, core, latency_ms, metadata)

    def _typesafe(self, model: Dict[str, Any], state: Dict[str, str], questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        key = self.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise InputError("선택한 원격 모델의 API 키가 설정되지 않았습니다.")
        body = {"model": model["wire_model"], "state": state, "questions": questions}
        headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
        started = time.perf_counter()
        response = self.transport("https://api.typesafe.ai/v1/systemone", headers, body, REMOTE_TIMEOUT_SECONDS)
        latency_ms = (time.perf_counter() - started) * 1000
        return normalize_result(model, response, latency_ms)

    def predict(self, browser_payload: Any) -> Dict[str, Any]:
        model = resolve_model(browser_payload.get("model_id") if isinstance(browser_payload, dict) else None)
        state, questions, presentation = build_payload(browser_payload)
        check_capability(model, questions)
        presentation["state"] = state
        presentation["questions"] = questions
        if model["provider"] == "local":
            normalized = self._local(model, state, questions)
        elif model["provider"] == "decisions":
            normalized = self._decisions(model, state, questions)
        else:
            normalized = self._typesafe(model, state, questions)
        return public_result(normalized, next(iter(questions)), presentation)


def make_handler(service: LayaService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "LayaLocal/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            # Avoid putting submitted text in terminal logs; keep only method/path/status.
            sys.stderr.write("%s %s\n" % (self.command, self.path))

        def send_json(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/":
                body = HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/health":
                self.send_json(HTTPStatus.OK, {"ready": service.ready, "loaded": service.router.loaded if service.router else [], "device": service.device, "error": service.error, "models": service.catalog_for_browser()})
            else:
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "없는 경로입니다."})

        def do_POST(self) -> None:
            if self.path != "/api/predict":
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "없는 경로입니다."})
                return
            length = self.headers.get("Content-Length")
            try:
                size = int(length) if length is not None else -1
            except ValueError:
                size = -1
            if size < 0 or size > MAX_BODY_BYTES:
                self.send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "요청 본문은 %dKB 이하여야 합니다." % (MAX_BODY_BYTES // 1024)})
                return
            try:
                payload = json.loads(self.rfile.read(size).decode("utf-8"))
                self.send_json(HTTPStatus.OK, service.predict(payload))
            except UnicodeDecodeError:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": "요청 본문은 UTF-8 JSON이어야 합니다."})
            except json.JSONDecodeError:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": "요청 본문이 올바른 JSON이 아닙니다."})
            except InputError as exc:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except UpstreamError as exc:
                self.send_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
            except Exception as exc:  # Do not expose a local traceback to the browser.
                sys.stderr.write("prediction error: %r\n" % (exc,))
                self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "예측 중 오류가 발생했습니다. 서버 터미널 로그를 확인해 주세요."})

    return Handler


def self_test() -> None:
    """Exercise schema/adapters/UI without importing Laya or using the network."""
    choice = {"model_id": "decisions-typesafe-jev-1-13", "text": "결제가 두 번 됐습니다.", "instructions": "담당 부서는?", "type": "choice", "criteria": "결제: 결제 또는 환불\n기술지원: 오류 또는 장애"}
    state, questions, shown = build_payload(choice)
    assert state == {"input": "결제가 두 번 됐습니다."}
    assert questions["judgement"]["criteria"] == {"결제": "결제 또는 환불", "기술지원": "오류 또는 장애"}
    assert build_payload({**choice, "type": "noul", "false_criterion": "아니다", "true_criterion": "맞다", "threshold": "0.5"})[2]["threshold"] == 0.5
    assert build_payload({**choice, "type": "score", "criteria": "낮음: 약함\n높음: 강함"})[1]["judgement"]["criteria"] == ["낮음: 약함", "높음: 강함"]
    with tempfile.TemporaryDirectory() as temporary_root:
        temporary_dotenv = Path(temporary_root) / ".env"
        temporary_dotenv.write_text(
            "DECISION_ORG_API_KEY=decisions-dotenv-test-only\n"
            "TYPESAFE_API_KEY=typesafe-dotenv-test-only\n"
            "OTHER_SETTING=must-not-load\n",
            encoding="utf-8",
        )
        dotenv_environment: Dict[str, str] = {}
        load_project_dotenv(dotenv_environment, temporary_dotenv)
        assert dotenv_environment["DECISION_ORG_API_KEY"] == "decisions-dotenv-test-only"
        assert dotenv_environment["TYPESAFE_API_KEY"] == "typesafe-dotenv-test-only"
        assert "OTHER_SETTING" not in dotenv_environment
        dotenv_catalog = LayaService("mps", environ=dotenv_environment).catalog_for_browser()
        assert all(next(item for item in dotenv_catalog if item["provider"] == provider)["configured"] is True
                   for provider in ("decisions", "typesafe"))
        exported_environment = {"DECISION_ORG_API_KEY": "", "TYPESAFE_API_KEY": ""}
        load_project_dotenv(exported_environment, temporary_dotenv)
        assert exported_environment["DECISION_ORG_API_KEY"] == ""
        assert exported_environment["TYPESAFE_API_KEY"] == ""
    for broken in (
        {**choice, "text": ""},
        {**choice, "criteria": "결제: a\n결제: b"},
        {**choice, "type": "score", "criteria": "낮음: 같은 기준\n높음: 같은 기준"},
    ):
        try:
            build_payload(broken)
        except InputError:
            pass
        else:
            raise AssertionError("invalid payload was accepted")
    captured: List[Tuple[str, Dict[str, str], Dict[str, Any], float]] = []
    fixtures = [
        {"code": 0, "data": {"requestId": "request-1", "creditsUsed": 1, "historySaved": True, "result": {"model": "typesafe/jev-1.13", "answers": {"judgement": {"type": "choice", "choice": "결제", "probabilities": {"결제": .8, "기술지원": .2}, "confidence": .6}}, "usage": {"input_tokens": 9}, "elapsedMs": 3}}},
        {"code": 0, "data": {"requestId": "request-2", "creditsUsed": 1, "historySaved": False, "result": {"model": "typesafe/jev-1.13", "answers": {"judgement": {"type": "noul", "noul": .8}}, "usage": {"input_tokens": 10}, "elapsedMs": 4}}},
        {"code": 0, "data": {"requestId": "request-3", "creditsUsed": 1, "historySaved": False, "result": {"model": "typesafe/jev-1.13", "answers": {"judgement": {"type": "score", "score": .7, "legend": {"0": "낮음", "1": "높음"}, "probabilities": {"0": .3, "1": .7}, "confidence": .4}}, "usage": {"input_tokens": 11}, "elapsedMs": 5}}},
        {"model": "jev-1.13.0", "answers": {"judgement": {"type": "choice", "choice": "결제", "probabilities": {"결제": .9, "기술지원": .1}, "confidence": .8}}, "usage": {"input_tokens": 12, "output_tokens": 4}},
    ]
    def fake_transport(url: str, headers: Dict[str, str], body: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        captured.append((url, headers, body, timeout))
        return fixtures.pop(0)
    secret = "decisions-not-for-browser-or-rendered-result"
    typesafe_secret = "typesafe-not-for-browser-or-rendered-result"
    service = LayaService("mps", transport=fake_transport, environ={
        "DECISION_ORG_API_KEY": secret,
        "TYPESAFE_API_KEY": typesafe_secret,
    })
    rendered = service.predict(choice)
    noul = service.predict({**choice, "type": "noul", "false_criterion": "아니다", "true_criterion": "맞다", "threshold": "0.5"})
    score = service.predict({**choice, "type": "score", "criteria": "낮음: 약함\n높음: 강함"})
    typesafe_rendered = service.predict({**choice, "model_id": "typesafe-jev-latest"})
    assert rendered["selected_label"] == "결제" and noul["sdk_confidence"] is None and score["expected_score"] == .7
    assert typesafe_rendered["model"] == {
        "requested_catalog_id": "typesafe-jev-latest", "provider": "typesafe",
        "requested_wire_model": "jev-latest", "resolved_model": "jev-1.13.0",
    }
    assert captured[0][0] == "https://decisions-api.org/v1/systemone" and captured[0][2]["model"] == "typesafe/jev-1.13"
    assert captured[0][2]["state"] == {"input": choice["text"]} and captured[0][2]["questions"] == questions
    assert set(captured[0][1]) == {"Authorization", "Content-Type"}
    assert captured[0][1]["Authorization"] == "Bearer " + secret
    assert captured[3][0] == "https://api.typesafe.ai/v1/systemone" and captured[3][2]["model"] == "jev-latest"
    assert captured[3][2]["state"] == {"input": choice["text"]} and captured[3][2]["questions"] == questions
    assert set(captured[3][1]) == {"Authorization", "Content-Type"}
    assert captured[3][1]["Authorization"] == "Bearer " + typesafe_secret
    assert all(secret not in json.dumps(value, ensure_ascii=False)
               and typesafe_secret not in json.dumps(value, ensure_ascii=False)
               for value in (service.catalog_for_browser(), rendered, noul, score, typesafe_rendered))
    labels = {item["id"]: item["select_label"] for item in service.catalog_for_browser()}
    assert labels == {
        "local-laya-multilingual": "(local) Laya Multilingual",
        "decisions-typesafe-jev-1-13": "(decisions-api.org) Jev (1.13)",
        "typesafe-jev-latest": "(typesafe.ai) Jev (latest)",
    }
    remote = next(item for item in service.catalog_for_browser() if item["provider"] == "decisions")
    assert "모델 주체 TypeSafe" in remote["detail"] and "wire ID typesafe/jev-1.13" in remote["detail"]
    typesafe_remote = next(item for item in service.catalog_for_browser() if item["provider"] == "typesafe")
    assert "TypeSafe 직접 서빙" in typesafe_remote["detail"] and "wire ID jev-latest" in typesafe_remote["detail"]
    # This reaches the accepted UI maximum yet exceeds Decisions' documented body limit
    # because validation and transport both use the same UTF-8 JSON bytes.
    wide_labels = "\n".join("%s%s:%s" % (chr(0xAC00 + index), "가" * (MAX_CRITERION_CHARS - 1), "나" * MAX_CRITERION_CHARS)
                            for index in range(MAX_OPTIONS))
    maximum_ui_choice = {**choice, "text": "입" * MAX_TEXT_CHARS, "instructions": "질" * MAX_INSTRUCTIONS_CHARS, "criteria": wide_labels}
    max_state, max_questions, _ = build_payload(maximum_ui_choice)
    max_wire_body = {"model": "typesafe/jev-1.13", "state": max_state, "questions": max_questions}
    assert len(wire_json_bytes(max_wire_body)) > DECISIONS_MAX_BODY_BYTES
    calls_before_oversize = len(captured)
    try:
        service.predict(maximum_ui_choice)
    except InputError as exc:
        assert "32 KiB" in str(exc)
    else:
        raise AssertionError("maximum UI choice was sent despite exceeding Decisions body bytes")
    assert len(captured) == calls_before_oversize
    presentation = {**shown, "state": state, "questions": questions}
    base_normalized = {"requested_catalog_id": "decisions-typesafe-jev-1-13", "provider": "decisions",
                       "requested_wire_model": "typesafe/jev-1.13", "resolved_model": "typesafe/jev-1.13",
                       "usage": {}, "provider_metadata": {}, "latency_ms": 0}
    for malformed in (
        {"type": "choice", "choice": "결제", "probabilities": {}},
        {"type": "choice", "choice": "결제", "probabilities": {"결제": "not-a-number"}},
        {"type": "score", "score": 0, "legend": {"0": "낮음"}, "probabilities": {}},
    ):
        try:
            public_result({**base_normalized, "answers": {"judgement": malformed}}, "judgement", presentation)
        except UpstreamError:
            pass
        else:
            raise AssertionError("malformed probability answer was accepted")
    no_key = LayaService("mps", transport=fake_transport, environ={})
    try:
        no_key.predict(choice)
    except InputError:
        pass
    else:
        raise AssertionError("remote model accepted without API key")
    unavailable = next(item for item in no_key.catalog_for_browser() if item["provider"] == "decisions")
    assert unavailable["configured"] is False and unavailable["available"] is False and unavailable["reason"] == "API 키 미설정"
    unavailable_typesafe = next(item for item in no_key.catalog_for_browser() if item["provider"] == "typesafe")
    assert unavailable_typesafe["configured"] is False and unavailable_typesafe["available"] is False and unavailable_typesafe["reason"] == "API 키 미설정"
    try:
        no_key.predict({**choice, "model_id": "typesafe-jev-latest"})
    except InputError:
        pass
    else:
        raise AssertionError("TypeSafe model accepted without API key")
    assert set(choice) >= {"model_id", "text", "instructions", "type"} and "model" not in choice and "endpoint" not in choice
    assert len(MODEL_CATALOG) == 3 and all(set(item["question_types"]) == {"choice", "noul", "score"} for item in MODEL_CATALOG.values())
    assert "System One 판정 실험" in HTML and "/api/predict" in HTML
    assert "function clearResult()" in HTML and "addEventListener('input',clearResult)" in HTML
    assert "id=\"field_name\"" not in HTML and "id=\"question_id\"" not in HTML
    assert "preset_choice" in HTML and "preset_noul" in HTML and "preset_score" in HTML
    assert "id=\"model\"" in HTML and "id=\"model_metadata\"" in HTML and "id=\"remote_warning\"" in HTML
    assert "(local) Laya Multilingual" in HTML and "m.select_label" in HTML
    assert "DECISION_ORG_API_KEY" not in HTML and "TYPESAFE_API_KEY" not in HTML and "Authorization" not in HTML
    print("self-test: payload, catalog, fake Decisions/TypeSafe adapters, secrecy, HTML routes OK (no model/network)")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", default=8766, type=int, help="local TCP port (default: 8766)")
    parser.add_argument("--device", default="mps", choices=("mps", "cpu"), help="Laya device (default: mps)")
    parser.add_argument("--self-test", action="store_true", help="validate API/HTML paths without importing or loading a model")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.port < 1 or args.port > 65535:
        parser.error("--port must be between 1 and 65535")
    # Deliberately run only for the server command: importing this module and --self-test
    # must not read or mutate the caller's project environment.
    load_project_dotenv()
    service = LayaService(args.device)
    try:
        # Bind before any expensive model construction; an occupied port must fail fast.
        server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    except OSError as exc:
        print("서버 bind 실패 (%s:%d): %s" % (args.host, args.port, exc), file=sys.stderr)
        return 1
    try:
        print("Laya multilingual 모델을 %s에 preload합니다…" % args.device, flush=True)
        service.start()
    except Exception as exc:
        service.error = str(exc)
        server.server_close()
        print("모델 preload 실패: %s" % exc, file=sys.stderr)
        return 1
    print("준비됨: http://%s:%d  (Ctrl-C로 종료)" % (args.host, args.port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버를 종료합니다.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
