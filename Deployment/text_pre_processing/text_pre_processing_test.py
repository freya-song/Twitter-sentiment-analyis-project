import unittest
import pytest
import text_pre_processing as tp

class Test(unittest.TestCase):
    def setup(self):
        return
    
    def test_init_1(self):
        pp = tp.PreProcessor(10, 100)
        exp_max_length_tweet = 10
        max_length_tweet = pp.max_length_tweet
        self.assertEquals(max_length_tweet, exp_max_length_tweet)

    def test_init_2(self):
        pp = tp.PreProcessor(10, 100)
        exp_max_length_dictionary = 100
        max_length_dictionary = pp.max_length_dictionary
        self.assertEquals(max_length_dictionary, exp_max_length_dictionary)

    def test_init_3(self):
        with pytest.raises(ValueError):
            tp.PreProcessor(-1, 100)
    
    def test_init_4(self):
        with pytest.raises(ValueError):
            tp.PreProcessor(1, -100)

    def test_init_5(self):
        pp = tp.PreProcessor(1)
        exp_dictionary_length = 400002
        dictionary_length = len(pp.emb_dict)
        self.assertEquals(dictionary_length, exp_dictionary_length)

    def test_method_clean_text_1(self):
        pp = tp.PreProcessor(10, 100)
        text = pp.clean_text("This is a simple test")
        exp_text = "This is a simple test"
        self.assertEquals(text, exp_text)
    
    def test_method_clean_text_2(self):
        pp = tp.PreProcessor(10, 100)
        text = pp.clean_text("https://stackoverflow.com/questions/23337471/how-to-properly-assert-that-an-exception-gets-raised-in-pytest This is a simple test")
        exp_text = "This is a simple test"
        self.assertEquals(text, exp_text)

    def test_method_clean_text_3(self):
        pp = tp.PreProcessor(10, 100)
        text = pp.clean_text("This is a https://stackoverflow.com/questions/23337471/how-to-properly-assert-that-an-exception-gets-raised-in-pytest simple test")
        exp_text = "This is a  simple test"
        self.assertEquals(text, exp_text)

    def test_method_clean_text_4(self):
        pp = tp.PreProcessor(10, 100)
        text = pp.clean_text("This is a simple test with some interesting \U0001F600\U0001F64F 其他语言或字符 within it.")
        exp_text = "This is a simple test with some interesting   within it."
        self.assertEquals(text, exp_text)

    def test_method_tokenize_text_1(self):
        pp = tp.PreProcessor(10, 100)
        text = pp.clean_text("This is a simple test with some interesting within it. https://stackoverflow.com/questions/23337471/how-to-properly-assert-that-an-exception-gets-raised-in-pytest")
        tokens = pp.tokenize_text(text)
        exp_tokens = ["This", "is", "a", "simple", "test", "with", "some", "interesting", "within", "it"]
        self.assertEquals(tokens, exp_tokens)
    
    def test_pad_sequence_1(self):
        pp = tp.PreProcessor(5, 100)
        pad = pp.pre_process("This is a simple test with some interesting \U0001F600\U0001F64F 其他语言或字符 within it. https://stackoverflow.com/questions/23337471/how-to-properly-assert-that-an-exception-gets-raised-in-pytest")
        exp_pad = [1, 16, 9, 1, 1]
        self.assertEquals(pad, exp_pad)
    
    def test_pad_sequence_2(self):
        pp = tp.PreProcessor(20, 100)
        pad = pp.pre_process("This is a simple test with some interesting \U0001F600\U0001F64F 其他语言或字符 within it. https://stackoverflow.com/questions/23337471/how-to-properly-assert-that-an-exception-gets-raised-in-pytest")
        exp_pad = [1, 16, 9, 1, 1, 19, 79, 1, 1, 22, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.assertEquals(pad, exp_pad)
