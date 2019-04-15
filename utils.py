def read_text(path):
    with open(path, 'r') as file:
        data = file.read()
    return data

def convert_to_int(data):
    chars = set(data)
    chars_as_int = dict([(x, i) for i, x in enumerate(chars)])
    data_as_int = [chars_as_int[x] for x in data]
    return data_as_int