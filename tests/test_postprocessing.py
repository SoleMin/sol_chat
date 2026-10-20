from chatbot.postprocessing import clean_chunk, clean_response


def test_clean_response():
    assert clean_response("  안녕<|im_end|>\n\n\n\n반가워 ") == "안녕\n\n반가워"


def test_clean_chunk_keeps_spaces():
    assert clean_chunk(" 안녕 </s>") == " 안녕 "
