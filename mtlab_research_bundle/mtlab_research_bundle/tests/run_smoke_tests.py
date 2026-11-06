from test_nltk_alignment import test_nltk_ibm4_ibm5
from test_phrase_based import test_phrase_pipeline
from test_nmt_models import test_nmt_architectures


def main():
    test_nltk_ibm4_ibm5()
    test_phrase_pipeline()
    test_nmt_architectures()
    print("All bundled smoke tests passed.")


if __name__ == "__main__":
    main()
