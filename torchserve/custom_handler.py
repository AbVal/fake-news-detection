import torch
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Transformer model from path {model_dir} loaded successfully")

        self.initialized = True

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        preprocessed_data = data[0].get("data")
        if preprocessed_data is None:
            preprocessed_data = data[0].get("body")

        preprocessed_data = preprocessed_data.decode('utf-8')

        preprocessed_data = self.tokenizer.encode_plus(
            preprocessed_data,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True
        )

        return preprocessed_data

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        model_output = self.model(
            model_input['input_ids'].to(self.device)
        )[0].argmax().item()
        logger.info("Model predicted: '%s'", model_output)

        return [model_output]

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        label_map = {0: "NEGATIVE", 1: "POSITIVE"}
        postprocess_output = [label_map[output] for output in inference_output]
        return postprocess_output

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)
