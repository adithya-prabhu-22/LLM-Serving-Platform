from tests.sanity.test_activations import run_all_tests as activations_tests
from tests.sanity.test_attention import run_all_tests as attention_tests
from tests.sanity.test_embeddings import run_all_tests as embeddings_tests
from tests.sanity.test_feedforward import run_all_tests as feedforward_tests
from tests.sanity.test_normalization import run_all_tests as normalization_tests
from tests.sanity.test_transformer_block import run_all_tests as transformer_block_tests
from tests.sanity.test_gpt import run_all_tests as gpt_tests
from tests.sanity.test_gpt_config import run_all_tests as gpt_config_tests
from tests.sanity.test_registry_service import run_all_tests as registry_tests
from tests.sanity.test_lifecycle_manager import run_all_tests as lifecycle_tests
from tests.sanity.test_model_loader import run_all_tests as model_loader_tests
from tests.sanity.test_model_loader_weights import run_all_tests as model_loader_weights_tests
from tests.sanity.test_text_generation import run_all_tests as text_generation_tests
from tests.sanity.test_inference_engine import run_all_tests as inference_engine_tests
from tests.sanity.test_inference_generation import run_all_tests as inference_generation_tests

def run_all():
    print("\n========== SANITY TEST SUITE ==========\n")
    activations_tests()
    attention_tests()
    embeddings_tests()
    feedforward_tests()
    normalization_tests()
    transformer_block_tests()
    gpt_tests()
    gpt_config_tests()
    registry_tests()
    lifecycle_tests()
    model_loader_tests()
    model_loader_weights_tests()
    text_generation_tests()
    inference_engine_tests()
    inference_generation_tests()
    print("\n========== ALL TESTS PASSED ==========\n")

if __name__ == "__main__":
    run_all()