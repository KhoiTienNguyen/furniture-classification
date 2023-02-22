import model_test
import model

test_model = model.Model()

def test_valid_output():
    result = test_model.predict('https://www.ikea.com/ca/en/images/products/hattefjaell-office-chair-with-armrests-smidig-black__1019087_pe831296_s5.jpg?f=xl')
    print(result)
    assert result in ['Chair', 'Bed', 'Sofa']

def test_invalid_input():
    try:
        assert test_model.predict('google.com')
        assert False
    except Exception:
        assert True