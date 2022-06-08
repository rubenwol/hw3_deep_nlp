import rstr
import string
import random

def generate_pos():
    return rstr.xeger(r'[1-9]{1,15}a{1,15}[1-9]{1,15}b{1,15}[1-9]{1,15}c{1,15}[1-9]{1,15}d{1,15}[1-9]{1,15}')
    # return rstr.xeger(r'[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+')
def generate_neg():
    return rstr.xeger(r'[1-9]{1,15}a{1,15}[1-9]{1,15}c{1,15}[1-9]{1,15}b{1,15}[1-9]{1,15}d{1,15}[1-9]{1,15}')
    # return rstr.xeger(r'[1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+')

def generate_pos_dataset(data_size=500):
    data = []
    for i in range(data_size):
        data.append(generate_pos())
    return data

def generate_neg_dataset(data_size=500):
    data = []
    for i in range(data_size):
        data.append(generate_neg())
    return data


##########################################################################
def generate_palindrome():
    random_seq = rstr.rstr(string.ascii_letters + string.digits, 1, 50)
    return random_seq + random_seq[::-1]

def generate_random_sequence():
    random_seq = rstr.rstr(string.ascii_letters + string.digits, 1, 100)
    return random_seq

def generate_palindrome_dataset(data_size=500):
    data = []
    for i in range(data_size):
        data.append(generate_palindrome())
    return data

def generate_random_seq_dataset(data_size=500):
    data = []
    for i in range(data_size):
        data.append(generate_random_sequence())
    return data
#########################################################################

def generate_ww_language():
    random_seq = rstr.rstr(string.ascii_letters + string.digits, 1, 50)
    return random_seq + random_seq

def generate_ww_dataset(data_size=500):
    data = []
    for i in range(data_size):
        data.append(generate_ww_language())
    return data

###########################################################################

def generate_0n_1n():
    a = rstr.rstr(string.ascii_letters, 1)
    b = rstr.rstr(string.ascii_letters, 1)
    while a == b:
        b = rstr.rstr(string.ascii_letters, 1)
    n_0 = random.randint(0, 50)
    n_1 = random.randint(0, 50)
    return a * n_0 + b * n_0 #+ '0' * n_0 + '1' * n_1

def generate_01_string():
    a = rstr.rstr(string.ascii_letters, 1)
    b = rstr.rstr(string.ascii_letters, 1)
    while a == b:
        b = rstr.rstr(string.ascii_letters, 1)
    n_0 = random.randint(0, 50)
    r_2 = list(range(1, n_0)) + list(range(n_0 + 1, 50))
    n_1 = random.choice(r_2)

    return a * n_0 + b * n_1



def generate_0n_1n_dataset(data_size=500):
    data = []
    for i in range(data_size):
        data.append(generate_0n_1n())
    return data

def generate_01_dataset(data_size=500):
    data = []
    for i in range(data_size):
        data.append(generate_01_string())
    return data



def data_to_file(fname, pos=True, data_size=500):
    if pos:
        data = generate_pos_dataset(data_size=data_size)
    else:
        data = generate_neg_dataset(data_size=data_size)
    with open(fname, 'w') as f:
        f.write('\n'.join(data))


if __name__ == '__main__':
    # data_pos = generate_pos_dataset()
    # data_neg = generate_neg_dataset()
    data_to_file('pos_examples', pos=True, data_size=500)
    data_to_file('neg_examples', pos=False, data_size=500)
    # data_to_file('data/pos_train', pos=True, data_size=2000)
    # data_to_file('data/neg_train', pos=False, data_size=5000)
    # data_to_file('data/pos_dev', pos=True, data_size=1000)
    # data_to_file('data/neg_dev', pos=False, data_size=1000)
    # data_to_file('data/pos_test', pos=True, data_size=1000)
    # data_to_file('data/neg_test', pos=False, data_size=1000)


