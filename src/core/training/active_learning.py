"""
Active Learning Module for CodeMind

Implements human-like learning where the model:
- Asks questions about things it doesn't know
- Learns from user feedback
- Improves continuously
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


@dataclass
class LearningQuestion:
    id: str
    question: str
    context: str
    difficulty: float
    uncertainty_score: float
    category: str
    language: str
    created_at: datetime = field(default_factory=datetime.now)
    answered: bool = False
    answer: Optional[str] = None


@dataclass
class LearningSession:
    id: str
    started_at: datetime
    questions_asked: int = 0
    questions_answered: int = 0
    knowledge_gained: int = 0
    categories_explored: Set[str] = field(default_factory=set)


class CuriosityEngine:
    def __init__(
        self, knowledge_base: Dict[str, Any], uncertainty_threshold: float = 0.3
    ):
        self.knowledge_base = knowledge_base
        self.uncertainty_threshold = uncertainty_threshold

        self.known_patterns: Set[str] = set()
        self.unknown_patterns: Set[str] = set()
        self.learning_history: List[Dict[str, Any]] = []

    def compute_uncertainty(self, pattern: str, examples: List[Dict]) -> float:
        if pattern in self.known_patterns:
            return 0.1

        if not examples:
            return 1.0

        seen_count = len(
            [
                ex
                for ex in examples
                if ex.get("pattern") == pattern and ex.get("learned", False)
            ]
        )

        if seen_count == 0:
            return 0.9
        elif seen_count < 3:
            return 0.7
        elif seen_count < 10:
            return 0.4
        else:
            return 0.1

    def find_curious_patterns(
        self, available_patterns: Dict[str, List[Dict]]
    ) -> List[Tuple[str, float]]:
        curious = []

        for pattern, examples in available_patterns.items():
            if pattern in self.known_patterns:
                continue

            uncertainty = self.compute_uncertainty(pattern, examples)

            if uncertainty >= self.uncertainty_threshold:
                curious.append((pattern, uncertainty))

        curious.sort(key=lambda x: x[1], reverse=True)

        return curious

    def generate_question(self, pattern: str, examples: List[Dict]) -> LearningQuestion:
        if not examples:
            raise ValueError(f"No examples for pattern: {pattern}")

        example = random.choice(examples)

        question_templates = {
            "function": [
                "Bu fonksiyon ne yapıyor ve nasıl çalışıyor?",
                "Bu fonksiyonu başka bir dille yazabilir misin?",
                "Bu fonksiyonu nasıl optimize edersin?",
            ],
            "class": [
                "Bu sınıfın amacı nedir?",
                "Bu sınıfa hangi metodları eklersin?",
                "Bu sınıfı nasıl genişletirsin?",
            ],
            "pattern": [
                "Bu tasarım desenini açıklar mısın?",
                "Bu pattern ne zaman kullanılır?",
                "Bu pattern'ın avantajları nelerdir?",
            ],
            "async": [
                "Bu asenkron akışta potansiyel bir race condition var mı?",
                "Bu işlemi daha verimli bir asenkron pattern ile nasıl yazarsın?",
                "Burada 'await' kullanımı doğru mu, yoksa paralel çalıştırmak daha mı iyi?",
            ],
            "security": [
                "Bu kodda bir güvenlik açığı (örn. SQLi, XSS) görüyor musun?",
                "Kullanıcı girdisini burada nasıl daha güvenli bir şekilde sanitize edersin?",
                "Bu API endpoint'ini yetkisiz erişime karşı nasıl korursun?",
            ],
            "performance": [
                "Bu algoritmanın zaman karmaşıklığı (Big O) nedir?",
                "Bellek kullanımını azaltmak için bu kodu nasıl optimize edersin?",
                "Bu döngüyü daha hızlı çalışacak şekilde refactor edebilir misin?",
            ],
            "general": [
                "Bu kod ne yapıyor?",
                "Bu kodu nasıl iyileştirirsin?",
                "Bu kodu başka bir dille yazar mısın?",
            ],
        }

        category = example.get("category", "general")
        templates = question_templates.get(category, question_templates["general"])

        question_text = random.choice(templates)

        context = example.get("user", example.get("code", ""))
        if len(context) > 200:
            context = context[:200] + "..."

        uncertainty = self.compute_uncertainty(pattern, examples)

        return LearningQuestion(
            id=f"q_{pattern}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            question=question_text,
            context=context,
            difficulty=uncertainty,
            uncertainty_score=uncertainty,
            category=category,
            language=example.get("language", "python"),
        )

    def record_answer(
        self, question: LearningQuestion, answer: str, quality_score: float = 1.0
    ) -> None:
        self.learning_history.append(
            {
                "question_id": question.id,
                "pattern": question.category,
                "question": question.question,
                "answer": answer,
                "quality": quality_score,
                "timestamp": datetime.now().isoformat(),
            }
        )

        if quality_score >= 0.7:
            self.known_patterns.add(question.category)
        elif question.category in self.known_patterns:
            self.known_patterns.remove(question.category)


class ActiveLearner:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        knowledge_base: Dict[str, List[Dict]],
        max_questions_per_session: int = 5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.knowledge_base = knowledge_base
        self.max_questions_per_session = max_questions_per_session

        self.curiosity = CuriosityEngine(knowledge_base)
        self.current_session: Optional[LearningSession] = None
        self.pending_questions: List[LearningQuestion] = []

    def start_session(self) -> LearningSession:
        self.current_session = LearningSession(
            id=f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            started_at=datetime.now(),
        )

        self.pending_questions = []

        return self.current_session

    def ask_question(self) -> Optional[LearningQuestion]:
        if self.current_session is None:
            self.start_session()

        if len(self.pending_questions) >= self.max_questions_per_session:
            return None

        curious_patterns = self.curiosity.find_curious_patterns(self.knowledge_base)

        if not curious_patterns:
            return None

        pattern, uncertainty = curious_patterns[0]
        examples = self.knowledge_base[pattern]

        question = self.curiosity.generate_question(pattern, examples)

        self.pending_questions.append(question)
        self.current_session.questions_asked += 1

        return question

    def receive_answer(
        self, question_id: str, answer: str, code_example: Optional[str] = None
    ) -> Dict[str, Any]:
        question = next(
            (q for q in self.pending_questions if q.id == question_id), None
        )

        if question is None:
            return {"error": "Question not found"}

        question.answered = True
        question.answer = answer

        quality_score = self._evaluate_answer(answer, code_example)

        self.curiosity.record_answer(question, answer, quality_score)

        learning_item = {
            "user": question.context,
            "assistant": answer,
            "pattern": question.category,
            "language": question.language,
            "learned": True,
            "quality": quality_score,
        }

        if question.category in self.knowledge_base:
            self.knowledge_base[question.category].append(learning_item)
        else:
            self.knowledge_base[question.category] = [learning_item]

        if self.current_session:
            self.current_session.questions_answered += 1
            self.current_session.knowledge_gained += 1
            self.current_session.categories_explored.add(question.category)

        return {
            "quality_score": quality_score,
            "pattern_learned": question.category,
            "session_stats": {
                "questions_answered": self.current_session.questions_answered
                if self.current_session
                else 0,
                "knowledge_gained": self.current_session.knowledge_gained
                if self.current_session
                else 0,
            },
        }

    def _evaluate_answer(
        self, answer: str, code_example: Optional[str] = None
    ) -> float:
        score = 0.5

        if len(answer) > 50:
            score += 0.1

        if "```" in answer:
            score += 0.2

        if code_example and len(code_example) > 20:
            score += 0.2

        keywords = ["fonksiyon", "function", "class", "sınıf", "method", "metod"]
        if any(kw in answer.lower() for kw in keywords):
            score += 0.1

        return min(1.0, score)

    def end_session(self) -> Dict[str, Any]:
        if self.current_session is None:
            return {"error": "No active session"}

        stats = {
            "session_id": self.current_session.id,
            "duration_seconds": (
                datetime.now() - self.current_session.started_at
            ).total_seconds(),
            "questions_asked": self.current_session.questions_asked,
            "questions_answered": self.current_session.questions_answered,
            "knowledge_gained": self.current_session.knowledge_gained,
            "categories_explored": list(self.current_session.categories_explored),
        }

        self.current_session = None
        self.pending_questions = []

        return stats

    def get_learning_progress(self) -> Dict[str, Any]:
        total_patterns = len(self.knowledge_base)
        known_patterns = len(self.curiosity.known_patterns)

        return {
            "total_patterns": total_patterns,
            "known_patterns": known_patterns,
            "learning_progress": known_patterns / total_patterns
            if total_patterns > 0
            else 0,
            "unknown_patterns": total_patterns - known_patterns,
        }


class DifficultyAssessor:
    def __init__(self, initial_difficulty: float = 0.5):
        self.current_difficulty = initial_difficulty
        self.history: List[Dict[str, Any]] = []

    def assess_code_difficulty(self, code: str) -> float:
        difficulty = 0.0

        lines = code.strip().split("\n")
        if len(lines) > 20:
            difficulty += 0.2
        elif len(lines) > 10:
            difficulty += 0.1

        complex_keywords = [
            "class",
            "async",
            "await",
            "decorator",
            "@",
            "lambda",
            "generator",
            "yield",
            "recursion",
        ]
        for kw in complex_keywords:
            if kw in code:
                difficulty += 0.05

        if code.count("(") > 5:
            difficulty += 0.1
        if code.count("[") > 3:
            difficulty += 0.05

        return min(1.0, difficulty)

    def adapt_difficulty(self, success: bool, time_taken: float) -> float:
        if success:
            if time_taken < 30:
                self.current_difficulty = min(1.0, self.current_difficulty + 0.1)
            elif time_taken > 120:
                self.current_difficulty = max(0.1, self.current_difficulty - 0.05)
        else:
            self.current_difficulty = max(0.1, self.current_difficulty - 0.1)

        self.history.append(
            {
                "difficulty": self.current_difficulty,
                "success": success,
                "time_taken": time_taken,
            }
        )

        return self.current_difficulty

    def get_appropriate_examples(
        self, examples: List[Dict], count: int = 5
    ) -> List[Dict]:
        scored_examples = []

        for ex in examples:
            code = ex.get("user", ex.get("code", ""))
            diff = self.assess_code_difficulty(code)

            distance = abs(diff - self.current_difficulty)
            scored_examples.append((distance, ex))

        scored_examples.sort(key=lambda x: x[0])

        return [ex for _, ex in scored_examples[:count]]


class ContinuousLearningPipeline:
    def __init__(
        self, model: Any, tokenizer: Any, data_path: str, save_path: str = "checkpoints"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.data_path = Path(data_path)
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.knowledge_base = self._load_knowledge_base()

        self.active_learner = ActiveLearner(model, tokenizer, self.knowledge_base)

        self.difficulty_assessor = DifficultyAssessor()

    def _load_knowledge_base(self) -> Dict[str, List[Dict]]:
        kb: Dict[str, List[Dict]] = {}

        for file_path in self.data_path.glob("**/*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                pattern = file_path.stem

                for item in data:
                    if "pattern" not in item:
                        item["pattern"] = pattern

                kb[pattern] = data
            except Exception:
                continue

        return kb

    def interactive_learning_round(self, get_answer_func: callable) -> Dict[str, Any]:
        self.active_learner.start_session()

        results = []

        while True:
            question = self.active_learner.ask_question()

            if question is None:
                break

            answer = get_answer_func(question)

            if answer is None:
                break

            result = self.active_learner.receive_answer(
                question.id, answer.get("text", ""), answer.get("code")
            )
            results.append(result)

        session_stats = self.active_learner.end_session()

        return {"session": session_stats, "learning_results": results}

    def save_progress(self) -> None:
        progress = {
            "known_patterns": list(self.active_learner.curiosity.known_patterns),
            "learning_history": self.active_learner.curiosity.learning_history,
            "difficulty": self.difficulty_assessor.current_difficulty,
        }

        with open(
            self.save_path / "learning_progress.json", "w", encoding="utf-8"
        ) as f:
            json.dump(progress, f, ensure_ascii=False, indent=2, default=str)

        with open(self.save_path / "knowledge_base.json", "w", encoding="utf-8") as f:
            json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)

    def load_progress(self) -> None:
        progress_path = self.save_path / "learning_progress.json"
        if progress_path.exists():
            with open(progress_path, "r", encoding="utf-8") as f:
                progress = json.load(f)

            self.active_learner.curiosity.known_patterns = set(
                progress.get("known_patterns", [])
            )
            self.active_learner.curiosity.learning_history = progress.get(
                "learning_history", []
            )
            self.difficulty_assessor.current_difficulty = progress.get(
                "difficulty", 0.5
            )
