class singleModel(object):
    def __init__(self, configs):
        self.model = None
        # put your model configs here
        # self.train = config["train_file"]
        # ....
        
    def pred(feature):
        """
        return predictions: 0-1
        """
        pass

    def feature_extract():
        """
        extract feature from raw data
        """
        pass

    def train():
        pass

    def save():
        """
        save model
        """
        pass

    def load():
        """
        load model
        """
        pass


    def convert2onnx():
        """
        if inference time(except video model) is greater than 10ms on your machine:
            1. reduce model size
            2. use onnx https://github.com/onnx/tutorials
        """
        pass
    
    def process_one():
        """
        input: raw data 
        output: prediction
        """

   

