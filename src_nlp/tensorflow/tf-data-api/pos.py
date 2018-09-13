def load_data():
    word2idx = {'<pad>': 0}
    tag2idx = {'<pad>': 0}
    word_idx = 1
    tag_idx = 1
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for line in open('../temp/pos_train.txt'):
        line = line.rstrip()
        if line:
            word, tag, _ = line.split()
            if word not in word2idx:
                word2idx[word] = word_idx
                word_idx += 1
            x_train.append(word2idx[word])
            if tag not in tag2idx:
                tag2idx[tag] = tag_idx
                tag_idx += 1
            y_train.append(tag2idx[tag])

    word2idx['<unk>'] = word_idx

    for line in open('../temp/pos_test.txt'):
        line = line.rstrip()
        if line:
            word, tag, _ = line.split()
            if word in word2idx:
                x_test.append(word2idx[word])
            else:
                x_test.append(word_idx)
            y_test.append(tag2idx[tag])

    print("Vocab Size: %d | x_train: %d | x_test: %d" % (len(word2idx), len(x_train), len(x_test)))
    return x_train, y_train, x_test, y_test, len(word2idx), tag_idx, word2idx, tag2idx
