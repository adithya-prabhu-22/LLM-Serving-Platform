import torch

from core.models.embeddings import GPTEmbeddings


def test_embedding_forward():

    embeddings = GPTEmbeddings(
        vocab_size=50257,
        d_model=768,
        block_size=1024,
        dropout=0.1,
    )

    input_ids = torch.randint(
        low=0,
        high=50257,
        size=(2, 16),
    )

    output = embeddings(input_ids)

    assert output.shape == (
        2,
        16,
        768,
    )

    print("✓ Forward pass test passed")


def test_embedding_with_start_pos():

    embeddings = GPTEmbeddings(
        vocab_size=50257,
        d_model=768,
        block_size=1024,
    )

    input_ids = torch.randint(
        low=0,
        high=50257,
        size=(2, 8),
    )

    output = embeddings(
        input_ids,
        start_pos=100,
    )

    assert output.shape == (
        2,
        8,
        768,
    )

    print("✓ KV-cache position test passed")


def test_invalid_sequence_length():

    embeddings = GPTEmbeddings(
        vocab_size=50257,
        d_model=768,
        block_size=16,
    )

    input_ids = torch.randint(
        low=0,
        high=50257,
        size=(2, 20),
    )

    try:
        embeddings(input_ids)

        raise AssertionError(
            "Expected ValueError was not raised."
        )

    except ValueError:
        print("✓ Sequence length validation passed")


def test_invalid_input_shape():

    embeddings = GPTEmbeddings(
        vocab_size=50257,
        d_model=768,
        block_size=1024,
    )

    invalid_input = torch.randint(
        low=0,
        high=50257,
        size=(16,),
    )

    try:
        embeddings(invalid_input)

        raise AssertionError(
            "Expected ValueError was not raised."
        )

    except ValueError:
        print("✓ Input shape validation passed")


def run_all_tests():

    print("\nRunning GPTEmbeddings tests...\n")

    test_embedding_forward()
    test_embedding_with_start_pos()
    test_invalid_sequence_length()
    test_invalid_input_shape()

    print("\n✓ All GPTEmbeddings tests passed")


if __name__ == "__main__":
    run_all_tests()