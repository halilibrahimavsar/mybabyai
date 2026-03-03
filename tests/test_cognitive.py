import unittest
from src.core.cognitive.thought_node import ThoughtNode
from src.core.cognitive.reasoning_engine import ReasoningEngine
from src.core.cognitive.router import CognitiveRouter
from src.core.cognitive.modes import CognitiveMode

class TestMCTS(unittest.TestCase):
    def test_thought_node_uct(self):
        root = ThoughtNode("Question")
        root.visits = 10
        
        child1 = ThoughtNode("State A", "Action A", parent=root)
        child1.visits = 5
        child1.value = 4.0 # Avg: 0.8
        
        child2 = ThoughtNode("State B", "Action B", parent=root)
        child2.visits = 2
        child2.value = 0.2 # Avg: 0.1
        
        # Child 1 should have higher exploitation, Child 2 higher exploration.
        # Let's check calculation directly.
        c1_score = child1.uct_score(1.41)
        c2_score = child2.uct_score(1.41)
        
        # Child 1: 0.8 + 1.41 * sqrt(ln(10) / 5) = 0.8 + 1.41 * 0.678 = 1.75
        # Child 2: 0.1 + 1.41 * sqrt(ln(10) / 2) = 0.1 + 1.41 * 1.073 = 1.61
        self.assertTrue(c1_score > c2_score)

    def test_reasoning_engine_search(self):
        # Mock LLM and Evaluator
        def mock_lm_generate(context, n):
            # Always suggest simple numbered steps
            if "Step 1" not in context:
                return ["Step 1", "Bad Step 1"]
            return ["Step 2", "Bad Step 2"]
            
        def mock_evaluator(context, action):
            # Reward "Bad" steps lightly, "Step" steps highly
            if "Bad" in action:
                return 0.1
            return 0.9

        engine = ReasoningEngine(
            language_model_generate=mock_lm_generate,
            reward_evaluator=mock_evaluator,
            max_depth=2,
            simulations_per_step=10,
            exploration_weight=1.0,  # Lower for greedy behavior in test
            branching_factor=2
        )
        
        result_path = engine.search("Initial Prompt")
        
        # The engine should prefer "Step 1" over "Bad Step 1", etc.
        self.assertIn("Step 1", result_path)
        self.assertNotIn("Bad Step", result_path)

class TestCognitiveRouter(unittest.TestCase):
    def setUp(self):
        self.router = CognitiveRouter()

    def test_routing_system_1(self):
        config = self.router.route("Merhaba Dünya, nasılsın?")
        self.assertEqual(config.mode, CognitiveMode.SYSTEM_1)
        self.assertFalse(config.use_mcts)

    def test_routing_agent(self):
        config = self.router.route("Bana şu dosyayı oku ve terminalde çalıştır")
        self.assertEqual(config.mode, CognitiveMode.SYSTEM_2_AGENT)
        self.assertTrue(config.use_mcts)

    def test_routing_deepthink(self):
        config = self.router.route("Kuantum fiziği denklemindeki integral değerini calculate et.")
        self.assertEqual(config.mode, CognitiveMode.SYSTEM_2_DEEPTHINK)
        self.assertEqual(config.max_depth, 10)

    def test_routing_plan(self):
        config = self.router.route("Bir web sunucusu yazmak için adım adım plan oluştur.")
        self.assertEqual(config.mode, CognitiveMode.SYSTEM_2_PLAN)

if __name__ == '__main__':
    unittest.main()
