#!/usr/bin/env python3
"""
================================================================================
HIRA BigData Portal AI Chatbot
Solar 10.7B + LoRA (hira_lora_20251217_001)
================================================================================
포트: 8888
URL: /opnAI
접속: http://localhost:설정포트/proxy/8888/opnAI
================================================================================
"""

import os
import sys

# ========================================
# bitsandbytes 회피 (최상단 필수!)
# ========================================
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

sys.modules['bitsandbytes'] = None 
sys.modules['bitsandbytes.nn'] = None 
sys.modules['bitsandbytes.optim'] = None 
sys.modules['bitsandbytes.cuda_setup'] = None 
sys.modules['bitsandbytes.functional'] = None
# ========================================

import json
import time
import logging
import argparse
from datetime import datetime

import torch
from flask import Flask, request, jsonify, Response
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


# ========================================
# HTML 템플릿
# ========================================
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HIRA 빅데이터포털 AI</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 1200px;
            width: 100%;
            height: 90vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header h1 {
            font-size: 1.5rem;
            font-weight: 600;
        }

        .header .subtitle {
            font-size: 0.85rem;
            opacity: 0.9;
            margin-top: 4px;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(255, 255, 255, 0.2);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.85rem;
        }

        .status-dot {
            width: 10px;
            height: 10px;
            background: #00ff88;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.1); }
        }

        .main-content {
            display: flex;
            flex: 1;
            overflow: hidden;
        }

        .chat-section {
            flex: 1;
            display: flex;
            flex-direction: column;
            border-right: 1px solid #e0e0e0;
        }

        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
        }

        .message {
            display: flex;
            gap: 12px;
            margin-bottom: 20px;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            flex-direction: row-reverse;
        }

        .avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            flex-shrink: 0;
        }

        .message.assistant .avatar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        .message.user .avatar {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }

        .message-content {
            max-width: 70%;
        }

        .bubble {
            padding: 12px 16px;
            border-radius: 16px;
            line-height: 1.6;
            font-size: 0.95rem;
            word-wrap: break-word;
        }

        .message.assistant .bubble {
            background: white;
            border: 1px solid #e0e0e0;
            border-top-left-radius: 4px;
            color: #333;
        }

        .message.user .bubble {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-top-right-radius: 4px;
        }

        .message-meta {
            font-size: 0.75rem;
            color: #888;
            margin-top: 4px;
            padding: 0 4px;
        }

        .input-section {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
        }

        .input-container {
            display: flex;
            gap: 12px;
            align-items: flex-end;
        }

        .input-wrapper {
            flex: 1;
            position: relative;
        }

        #userInput {
            width: 100%;
            padding: 14px 18px;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            font-size: 1rem;
            resize: none;
            height: 52px;
            font-family: inherit;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        #userInput:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        #sendBtn {
            padding: 14px 28px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 12px;
            color: white;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            height: 52px;
        }

        #sendBtn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }

        #sendBtn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .sidebar {
            width: 360px;
            padding: 20px;
            background: #f8f9fa;
            overflow-y: auto;
        }

        .panel {
            background: white;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }

        .panel h3 {
            font-size: 0.9rem;
            color: #333;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 2px solid #667eea;
        }

        .param-group {
            margin-bottom: 12px;
        }

        .param-group label {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: #666;
            margin-bottom: 6px;
        }

        .param-group input[type="range"] {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: #e0e0e0;
            outline: none;
            -webkit-appearance: none;
        }

        .param-group input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            cursor: pointer;
        }

        .param-group input[type="number"] {
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            font-size: 0.9rem;
        }

        .example-btn {
            display: block;
            width: 100%;
            padding: 10px 12px;
            margin-bottom: 8px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            font-size: 0.85rem;
            color: #333;
            cursor: pointer;
            text-align: left;
            transition: all 0.2s;
        }

        .example-btn:hover {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: transparent;
            transform: translateX(4px);
        }

        .clear-btn {
            width: 100%;
            padding: 10px;
            background: #fff5f5;
            border: 1px solid #ffcccc;
            border-radius: 8px;
            color: #cc0000;
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.2s;
        }

        .clear-btn:hover {
            background: #ffe5e5;
        }

        .model-info {
            font-size: 0.75rem;
            color: #888;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 6px;
            margin-top: 12px;
            line-height: 1.6;
        }

        .typing-indicator {
            display: flex;
            gap: 4px;
            padding: 8px 0;
        }

        .typing-indicator span {
            width: 8px;
            height: 8px;
            background: #667eea;
            border-radius: 50%;
            animation: bounce 1.4s infinite;
        }

        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }

        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-8px); }
        }

        /* FAQ 카테고리 탭 */
        .faq-tabs {
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            margin-bottom: 12px;
        }

        .faq-tab {
            padding: 6px 10px;
            font-size: 0.7rem;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            background: #f8f9fa;
            cursor: pointer;
            transition: all 0.2s;
            white-space: nowrap;
        }

        .faq-tab:hover {
            background: #e9ecef;
        }

        .faq-tab.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: transparent;
        }

        .faq-list {
            max-height: 360px;
            overflow-y: auto;
            padding-right: 4px;
        }

        .faq-list::-webkit-scrollbar {
            width: 6px;
        }

        .faq-list::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 3px;
        }

        .faq-list::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 3px;
        }

        .faq-list::-webkit-scrollbar-thumb:hover {
            background: #a1a1a1;
        }

        .faq-btn {
            display: block;
            width: 100%;
            padding: 8px 10px;
            margin-bottom: 6px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            font-size: 0.8rem;
            color: #333;
            cursor: pointer;
            text-align: left;
            transition: all 0.2s;
            line-height: 1.4;
        }

        .faq-btn:hover {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: transparent;
        }

        .faq-count {
            font-size: 0.7rem;
            color: #888;
            margin-bottom: 8px;
        }

        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }

        .panel-header h3 {
            margin-bottom: 0;
            border-bottom: none;
            padding-bottom: 0;
        }

        .panel-toggle {
            font-size: 0.8rem;
            color: #888;
            transition: transform 0.2s;
        }

        .panel-toggle.collapsed {
            transform: rotate(-90deg);
        }

        .panel-content {
            overflow: hidden;
            transition: max-height 0.3s ease;
        }

        .panel-content.collapsed {
            max-height: 0 !important;
        }

        .panel-content.expanded {
            margin-top: 12px;
            padding-top: 8px;
            border-top: 2px solid #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>🏥 HIRA 빅데이터포털 AI</h1>
                <div class="subtitle">Solar 10.7B + HIRA LoRA | 보건의료빅데이터 전문 어시스턴트</div>
            </div>
            <div class="status-badge">
                <span class="status-dot"></span>
                <span>Online</span>
            </div>
        </div>

        <div class="main-content">
            <div class="chat-section">
                <div class="messages" id="messages">
                    <div class="message assistant">
                        <div class="avatar">🤖</div>
                        <div class="message-content">
                            <div class="bubble">
                                안녕하세요! HIRA 빅데이터포털 AI입니다. 🏥<br><br>
                                회원가입, 데이터 신청, IRB, 원격분석, CDM 등<br>
                                빅데이터개방시스템에 대해 궁금한 점을 물어보세요!
                            </div>
                        </div>
                    </div>
                </div>

                <div class="input-section">
                    <div class="input-container">
                        <div class="input-wrapper">
                            <textarea id="userInput" placeholder="질문을 입력하세요..."></textarea>
                        </div>
                        <button id="sendBtn">전송</button>
                    </div>
                </div>
            </div>

            <div class="sidebar">
                <div class="panel">
                    <div class="panel-header" onclick="togglePanel('paramPanel', 'paramToggle')">
                        <h3>⚙️ 생성 파라미터</h3>
                        <span class="panel-toggle collapsed" id="paramToggle">▼</span>
                    </div>
                    <div class="panel-content collapsed" id="paramPanel">
                    <div class="param-group">
                        <label>
                            <span>Temperature</span>
                            <span id="tempValue">0.7</span>
                        </label>
                        <input type="range" id="temperature" min="0.1" max="2.0" step="0.1" value="0.7">
                    </div>
                    <div class="param-group">
                        <label>
                            <span>Max Tokens</span>
                            <span id="maxTokensValue">256</span>
                        </label>
                        <input type="range" id="maxTokens" min="64" max="1024" step="64" value="256">
                    </div>
                    <div class="param-group">
                        <label>
                            <span>Top P</span>
                            <span id="topPValue">0.9</span>
                        </label>
                        <input type="range" id="topP" min="0.1" max="1.0" step="0.1" value="0.9">
                    </div>
                    </div>
                </div>

                <div class="panel">
                    <h3>💡 자주 묻는 질문</h3>
                    <div class="faq-tabs" id="faqTabs"></div>
                    <div class="faq-count" id="faqCount">전체 70개 질문</div>
                    <div class="faq-list" id="faqList"></div>
                </div>

                <div class="panel">
                    <button class="clear-btn" id="clearBtn">🗑️ 대화 지우기</button>
                    <div class="model-info">
                        <strong>Model:</strong> Solar 10.7B Instruct<br>
                        <strong>LoRA:</strong> hira_lora_20251217_001<br>
                        <strong>Eval Loss:</strong> 0.3910
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // 대화 히스토리
        let conversationHistory = [];

        // DOM 요소
        const messagesDiv = document.getElementById('messages');
        const userInput = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');
        const clearBtn = document.getElementById('clearBtn');
        const tempSlider = document.getElementById('temperature');
        const maxTokensSlider = document.getElementById('maxTokens');
        const topPSlider = document.getElementById('topP');

        // 슬라이더 값 표시 업데이트
        tempSlider.addEventListener('input', function() {
            document.getElementById('tempValue').textContent = this.value;
        });
        maxTokensSlider.addEventListener('input', function() {
            document.getElementById('maxTokensValue').textContent = this.value;
        });
        topPSlider.addEventListener('input', function() {
            document.getElementById('topPValue').textContent = this.value;
        });

        // FAQ 데이터 (70건, 7개 카테고리)
        const FAQ_CATEGORIES = [
            { id: 'account', name: '🔑 계정/권한', icon: '🔑' },
            { id: 'data', name: '📊 데이터/서비스', icon: '📊' },
            { id: 'apply', name: '📝 신청/절차', icon: '📝' },
            { id: 'irb', name: '📋 IRB/윤리', icon: '📋' },
            { id: 'cost', name: '💰 비용/결제', icon: '💰' },
            { id: 'remote', name: '💻 원격분석', icon: '💻' },
            { id: 'cdm', name: '🔬 CDM', icon: '🔬' }
        ];

        const FAQ_QUESTIONS = {
            account: [
                "회원가입은 어떻게 하나요?",
                "기관 회원과 개인 회원의 차이는 무엇인가요?",
                "기관 등록은 어떻게 하나요?",
                "기관 승인에는 보통 얼마나 걸리나요?",
                "담당자(관리자) 변경은 어떻게 하나요?",
                "아이디 또는 비밀번호를 잊어버렸어요. 어떻게 하나요?",
                "본인인증이 계속 실패해요. 어떻게 해결하나요?",
                "로그인이 안 돼요. 어떤 점을 확인해야 하나요?",
                "회원정보(이메일/휴대폰)는 어디서 수정하나요?",
                "회원 탈퇴는 어떻게 하나요?",
                "권한 신청은 어떻게 하나요?",
                "권한이 반려되면 어떻게 재신청하나요?"
            ],
            data: [
                "보건의료빅데이터개방시스템이 무엇인가요?",
                "어떤 종류의 데이터를 제공하나요?",
                "맞춤형 데이터와 표준형 데이터의 차이는 무엇인가요?",
                "데이터 제공 방식에는 어떤 것들이 있나요?",
                "데이터 제공 단위(건/기간/범위)는 어떻게 이해하면 되나요?",
                "데이터는 어떤 포맷으로 제공되나요?",
                "데이터 제공 시 개인정보는 어떻게 보호되나요?",
                "데이터 이용 목적에 제한이 있나요?",
                "데이터 신청 전 미리 확인할 자료가 있나요?",
                "포털에서 제공하는 공지사항/가이드는 어디서 보나요?"
            ],
            apply: [
                "데이터 신청 절차는 어떻게 되나요?",
                "신청서 작성은 어디에서 하나요?",
                "신청서 작성 시 주의할 점은 무엇인가요?",
                "신청 상태는 어디서 확인하나요?",
                "신청 내용을 수정하려면 어떻게 하나요?",
                "신청을 취소하려면 어떻게 하나요?",
                "필수 제출 서류는 무엇인가요?",
                "연구계획서에는 어떤 내용을 써야 하나요?",
                "공동연구자가 있을 때는 어떻게 신청하나요?",
                "연구책임자 변경은 가능한가요?",
                "연구기간 연장은 어떻게 하나요?",
                "추가분석(리비전) 신청은 어떻게 하나요?",
                "동일 주제로 재신청할 때 절차가 달라지나요?",
                "신청 반려 사유는 보통 무엇인가요?",
                "반려 후 재신청 시 가장 먼저 고칠 것은 무엇인가요?",
                "문의는 어디로 해야 하나요?"
            ],
            irb: [
                "IRB 승인이 꼭 필요한가요?",
                "IRB 승인서에는 어떤 정보가 포함되어야 하나요?",
                "IRB 면제인 경우에도 서류가 필요한가요?",
                "연구대상자 동의서가 필요한가요?",
                "기관생명윤리위원회(IRB) 관련 용어를 설명해 주세요.",
                "IRB 승인서 파일 형식에 제한이 있나요?",
                "IRB 승인 기간이 만료되면 어떻게 하나요?",
                "윤리심의 관련해서 자주 반려되는 포인트는 무엇인가요?"
            ],
            cost: [
                "데이터 이용 비용은 어떻게 산정되나요?",
                "결제는 언제 진행하나요?",
                "결제 수단은 무엇을 지원하나요?",
                "세금계산서 발행은 어떻게 하나요?",
                "견적서는 어디서 받을 수 있나요?",
                "결제 후 환불이 가능한가요?",
                "결제 오류가 나면 어떻게 해야 하나요?",
                "결제 담당자 정보는 어디서 입력하나요?",
                "비용 관련 문의는 어디로 해야 하나요?",
                "무료로 이용 가능한 데이터도 있나요?"
            ],
            remote: [
                "원격분석시스템은 무엇인가요?",
                "원격분석시스템은 어떻게 접속하나요?",
                "원격분석 계정 발급 절차는 어떻게 되나요?",
                "원격분석에서 사용할 수 있는 소프트웨어는 무엇인가요?",
                "원격분석에서 파일 업로드/다운로드는 어떻게 하나요?",
                "원격분석 접속이 끊기면 어떻게 해야 하나요?",
                "원격분석 이용 시간은 어떻게 되나요?",
                "원격분석 사용 중 오류가 나면 어떻게 문의하나요?"
            ],
            cdm: [
                "CDM 데이터는 어떻게 신청하나요?",
                "HIRA K-OMOP 데이터는 어떤 특징이 있나요?",
                "CDM 데이터에서 제공하는 테이블 범위는 어떻게 되나요?",
                "CDM 데이터 분석을 위한 기본 가이드는 어디서 보나요?",
                "CDM 데이터는 원격분석에서만 이용 가능한가요?",
                "CDM 관련 문의는 어디로 해야 하나요?"
            ]
        };

        let currentCategory = 'account';

        // FAQ 탭 렌더링
        function renderFaqTabs() {
            const tabsDiv = document.getElementById('faqTabs');
            tabsDiv.innerHTML = FAQ_CATEGORIES.map(cat => 
                '<button class="faq-tab' + (cat.id === currentCategory ? ' active' : '') + '" data-category="' + cat.id + '">' + cat.name + '</button>'
            ).join('');
            
            tabsDiv.querySelectorAll('.faq-tab').forEach(function(tab) {
                tab.addEventListener('click', function() {
                    currentCategory = this.getAttribute('data-category');
                    renderFaqTabs();
                    renderFaqList();
                });
            });
        }

        // FAQ 목록 렌더링
        function renderFaqList() {
            const listDiv = document.getElementById('faqList');
            const countDiv = document.getElementById('faqCount');
            const questions = FAQ_QUESTIONS[currentCategory];
            const catInfo = FAQ_CATEGORIES.find(c => c.id === currentCategory);
            
            countDiv.textContent = catInfo.name + ' ' + questions.length + '개 질문';
            
            listDiv.innerHTML = questions.map(q => 
                '<button class="faq-btn" data-question="' + q + '">' + q + '</button>'
            ).join('');
            
            listDiv.querySelectorAll('.faq-btn').forEach(function(btn) {
                btn.addEventListener('click', function() {
                    var question = this.getAttribute('data-question');
                    userInput.value = question;
                    userInput.focus();
                });
            });
        }

        // FAQ 초기화
        renderFaqTabs();
        renderFaqList();

        // 패널 토글 함수
        function togglePanel(panelId, toggleId) {
            const panel = document.getElementById(panelId);
            const toggle = document.getElementById(toggleId);
            panel.classList.toggle('collapsed');
            panel.classList.toggle('expanded');
            toggle.classList.toggle('collapsed');
        }

        // 대화 지우기
        clearBtn.addEventListener('click', function() {
            conversationHistory = [];
            messagesDiv.innerHTML = '<div class="message assistant"><div class="avatar">🤖</div><div class="message-content"><div class="bubble">안녕하세요! HIRA 빅데이터포털 AI입니다. 🏥<br><br>회원가입, 데이터 신청, IRB, 원격분석, CDM 등<br>빅데이터개방시스템에 대해 궁금한 점을 물어보세요!</div></div></div>';
        });

        // Enter 키 처리
        userInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // 전송 버튼
        sendBtn.addEventListener('click', function() {
            sendMessage();
        });

        // 메시지 전송
        async function sendMessage() {
            var message = userInput.value.trim();
            if (!message) return;

            sendBtn.disabled = true;
            userInput.value = '';

            // 사용자 메시지 추가
            addMessage('user', message);

            // 타이핑 표시
            var typingId = showTyping();

            try {
                var response = await fetch('/proxy/8888/opnAI/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        message: message,
                        history: conversationHistory,
                        temperature: parseFloat(tempSlider.value),
                        max_tokens: parseInt(maxTokensSlider.value),
                        top_p: parseFloat(topPSlider.value)
                    })
                });

                var data = await response.json();
                hideTyping(typingId);

                if (data.error) {
                    addMessage('assistant', '오류가 발생했습니다: ' + data.error);
                } else {
                    addMessage('assistant', data.response, data.tokens, data.elapsed);
                    conversationHistory.push({
                        user: message,
                        assistant: data.response
                    });
                    if (conversationHistory.length > 5) {
                        conversationHistory = conversationHistory.slice(-5);
                    }
                }
            } catch (error) {
                hideTyping(typingId);
                addMessage('assistant', '네트워크 오류가 발생했습니다: ' + error.message);
            }

            sendBtn.disabled = false;
            userInput.focus();
        }

        // 메시지 추가
        function addMessage(role, content, tokens, elapsed) {
            var div = document.createElement('div');
            div.className = 'message ' + role;

            var avatar = role === 'user' ? '👤' : '🤖';
            var meta = '';
            if (tokens && elapsed) {
                meta = '<div class="message-meta">⏱ ' + elapsed + '초 | ' + tokens + ' tokens</div>';
            }

            var formattedContent = content.split('\\n').join('<br>');

            div.innerHTML = '<div class="avatar">' + avatar + '</div><div class="message-content"><div class="bubble">' + formattedContent + '</div>' + meta + '</div>';

            messagesDiv.appendChild(div);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        // 타이핑 표시
        function showTyping() {
            var div = document.createElement('div');
            div.className = 'message assistant';
            div.id = 'typing-' + Date.now();
            div.innerHTML = '<div class="avatar">🤖</div><div class="message-content"><div class="bubble"><div class="typing-indicator"><span></span><span></span><span></span></div></div></div>';
            messagesDiv.appendChild(div);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            return div.id;
        }

        // 타이핑 숨기기
        function hideTyping(id) {
            var el = document.getElementById(id);
            if (el) el.remove();
        }
    </script>
</body>
</html>
'''


# ========================================
# 모델 서버
# ========================================
class HIRAModelServer:
    SYSTEM_PROMPT = "You are a helpful AI assistant for HIRA BigData Portal. Please respond in the same language as the user's question."
    
    def __init__(self, base_model_path, lora_path):
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        logger.info("=" * 60)
        logger.info("HIRA AI 모델 로드")
        logger.info("  베이스: %s", self.base_model_path)
        logger.info("  LoRA: %s", self.lora_path)
        logger.info("=" * 60)
        
        # 토크나이저
        logger.info("[1/3] 토크나이저 로드...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("  완료")
        
        # 베이스 모델
        logger.info("[2/3] 베이스 모델 로드 (1-2분 소요)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map={"": 0},
            trust_remote_code=True,
            local_files_only=True,
        )
        logger.info("  완료")
        
        # LoRA 어댑터
        logger.info("[3/3] LoRA 어댑터 적용 및 병합...")
        self.model = PeftModel.from_pretrained(
            self.model,
            self.lora_path,
            local_files_only=True,
        )
        self.model = self.model.merge_and_unload()
        logger.info("  LoRA 병합 완료")
        
        self.model.eval()
        
        logger.info("=" * 60)
        logger.info("모델 로드 완료!")
        logger.info("=" * 60)
    
    def format_prompt(self, user_message, history=None):
        prompt = "### System:\n" + self.SYSTEM_PROMPT + "\n\n"
        
        if history:
            for h in history[-3:]:
                prompt += "### User:\n" + h['user'] + "\n\n### Assistant:\n" + h['assistant'] + "\n\n"
        
        prompt += "### User:\n" + user_message + "\n\n### Assistant:\n"
        return prompt
    
    @torch.inference_mode()
    def generate(self, message, history=None, max_tokens=256, temperature=0.7, top_p=0.9):
        prompt = self.format_prompt(message, history)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)
        
        input_len = inputs["input_ids"].shape[1]
        
        start_time = time.time()
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )
        
        elapsed = time.time() - start_time
        
        generated_ids = outputs[0][input_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Stop sequences 처리
        for stop in ["### User:", "### System:", "### Assistant:"]:
            if stop in response:
                response = response.split(stop)[0]
        
        response = response.strip()
        
        return {
            "response": response,
            "tokens": len(generated_ids),
            "elapsed": round(elapsed, 2),
        }


# 전역 모델 인스턴스
model_server = None


# ========================================
# Flask 라우트
# ========================================
@app.route('/opnAI')
@app.route('/opnAI/')
def index():
    return Response(HTML_TEMPLATE, mimetype='text/html')


@app.route('/opnAI/health')
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model_server is not None and model_server.model is not None,
    })


@app.route('/opnAI/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        message = data.get('message', '')
        history = data.get('history', [])
        
        if not message:
            return jsonify({"error": "메시지가 없습니다."})
        
        result = model_server.generate(
            message=message,
            history=history,
            max_tokens=data.get('max_tokens', 256),
            temperature=data.get('temperature', 0.7),
            top_p=data.get('top_p', 0.9),
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error("Generate error: %s", e)
        return jsonify({"error": str(e)})


# ========================================
# Main
# ========================================
def main():
    global model_server
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--base_model", default="./model/SOLAR-10.7B-Instruct-v1.0")
    parser.add_argument("--lora_model", default="./outputs/hira_lora_20251217_001/final_model")
    args = parser.parse_args()
    
    # 모델 로드
    model_server = HIRAModelServer(args.base_model, args.lora_model)
    model_server.load_model()
    
    # 서버 시작
    logger.info("")
    logger.info("=" * 60)
    logger.info("Flask 서버 시작")
    logger.info("  포트: %s", args.port)
    logger.info("  접속: http://0.0.0.0:0000/proxy/%s/opnAI", args.port)  # localhost
    logger.info("=" * 60)
    
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()