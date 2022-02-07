
def pprint_sample(_dict):
    for k, v in _dict.items():
        try:
            dtype = v.dtype
        except:
            dtype = type(v)

        try:
            shape = v.shape
        except:
            shape = ""

        try:
            if (v.numel() < 10):
                value = v
            else:
                value = ""
        except:
            value = ""

        try:
            if (v.shape == ()) or (v.shape == (1,)):
                value = v
        except:
            pass

        print(f"{k:20} {str(type(v)):50} {str(dtype):20} {str(shape):30} {str(value)}")