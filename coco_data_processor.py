def transform_batch(batch, processor):
    """
    input batch: tuple with the images and a list of tuples of sentences.
    the lenght of the list is the number of sentences for an image.
    the length of the tuple is the batch size.

    output batch: a tensor with for each image one of the sentences randomly chosen.
    the first dim is the batchsize. second dim is the sentence length.
    the sentences are padded with zeros and prefixed and post fixed with the
    START and END token. The words are transformed to indices.
    """
    sent_lengths = []
    longest = -1
    images,captions = batch
    trans_images = None
    trans_captions = []
    repeat_size = images[0].size()
#     repeat_size[0] *= 5
    print(repeat_size)
    print(images.size())
    for sample_num in range(len(captions[0])):
        number_of_captions = len(captions)
        if trans_images is None:
            print(images[sample_num].size())
            trans_images = images[sample_num].repeat([5,3,224,224])
            print(trans_images.size())
        else:
            result = torch.cat([trans_images, images[sample_num].repeat(repeat_size)],0)
        for sentnum in range(number_of_captions):
            s = [START] + wordpunct_tokenize(captions[sentnum][sample_num].lower()) + [END]
            l = len(s)
            trans_captions.append(s)
            sent_lengths.append(l)
            if longest < l:
                longest = l

    final_images = np.array(trans_images)
    final_images = torch.from_numpy(final_images).type(torch.LongTensor).to(device)
    final_captions = np.zeros((len(trans_captions), longest))
    for i,s in enumerate(trans_captions):
        final_captions[i,:len(s)] = np.array([processor.w2i[w] for w in s])
    batch = torch.from_numpy(trans_batch).type(torch.LongTensor).to(device)
    sent_lengths = torch.FloatTensor(sent_lengths).to(device)
    return batch, sent_lengths


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
temp_data = CocoCaptions(root = '/home/victor/coco/images/train2014/',
                         annFile = '/home/victor/coco/annotations/captions_train2014.json',
                         transform=transforms.ToTensor())
train_data = CocoCaptions(root = '/home/victor/coco/images/train2014/',
                          annFile = '/home/victor/coco/annotations/captions_train2014.json',
                          transform=transform)
val_data = CocoCaptions(root = '/home/victor/coco/images/val2014/',
                        annFile = '/home/victor/coco/annotations/captions_val2014.json',
                        transform=transform)