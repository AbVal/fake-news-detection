import argparse
import logging
import pandas as pd
from src.models import fetch_distilbert_tokenizer, load_trained_model

from transformers import pipeline

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="Path to model", default="model")
    parser.add_argument("--input_path",
                        help="Path to input data",
                        default="input.csv")
    parser.add_argument("--output_path",
                        help="Path to output data",
                        default="output.csv")

    args = parser.parse_args()

    model = load_trained_model(args.model_path)
    tokenizer = fetch_distilbert_tokenizer()

    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)

    input_data = pd.read_csv(args.input_path)

    texts = input_data["text"].astype(str).tolist()

    logger.info("Running predictions")
    predictions = pipe(texts, truncation=True)

    logger.info("Parsing results")
    results = []
    for i, pred in enumerate(predictions):

        results.append({
            'text': texts[i][:100],
            'prediction': pred['label'],
            'score': float(pred['score'])
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_path, index=False)
    logger.info(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
