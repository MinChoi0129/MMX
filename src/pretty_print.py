def shprint(*obj: object, **kwargs):
    isException = kwargs.get("isException", False)
    n = len(obj)
    print("---------------------------------" * n)
    for o in obj:
        try:
            print(o.shape, end=" | ")
        except:
            print(o, end=" | ")
    print()
    print("---------------------------------" * n)
    if isException:
        raise Exception("Pretty print done.")
