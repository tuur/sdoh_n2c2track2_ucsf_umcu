# -*- coding: utf-8 -*-

import random, os, shutil, numpy, torch, pandas, time, datetime, re, argparse, pickle
from collections import Counter
from torch import nn
from brat_scoring.corpus import Corpus
from brat_scoring.constants import EXACT, LABEL, OVERLAP
from brat_scoring.scoring import score_brat_sdoh, score_docs, micro_average_subtypes_csv
from brat_scoring.constants_sdoh import LABELED_ARGUMENTS as SDOH_LABELED_ARGUMENTS
from gensim.models import KeyedVectors

random.seed(1)


def load_model(path=None):
    print ('loading model', path)
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_event_trigger_info_from_bio_tags(bio_tags):
    events = []
    inevent = False
    start = 0
    etype = ''
    for i, tag in enumerate(bio_tags):
        if (inevent and tag[0] in ['O', 'B']):
            events += [(etype, start, i)]  # close previous event
        if tag[0] == 'B':
            start = i
            etype = tag[2:]
            inevent = True
            if i == len(bio_tags) - 1:
                events += [(etype, start, i + 1)]  # close previous event
        if tag[0] == 'O':
            inevent = False
    return events


class sdoh_model(nn.Module):
    def __init__(self, conflate_digits=True, lowercase=True, unk1_prob=0.5, wemb_size=200, rnn_dim=50, rnn_type='LSTM', subtask_weight=1, dropout=0.5, relus=1, emb_dropout=0, divide_subtask_networks=0, rnn_num_layers=1):
        super().__init__()
        self.word_emb_dim = wemb_size
        self.pretrained_embs=0
        self.rnn_dim = rnn_dim
        self.rnn_type = rnn_type
        self.rnn_num_layers = rnn_num_layers
        self.dropout = dropout
        self.emb_dropout=emb_dropout
        self.unk = '<UNK>'
        self.conflate_digits=conflate_digits
        self.lowercase=lowercase
        self.unk1_prob = unk1_prob
        self.wordsoffreq1 = set()
        self.subtask_weight=subtask_weight
        self.divide_subtask_networks=divide_subtask_networks
        self.relus=relus
        self.subtask_layers = nn.ModuleDict()
        self.subtask_wembs = nn.ModuleDict()
        self.subtask_rnns = nn.ModuleDict()
        self.word_embs = None
        self.rnn=None
        self.vocab={}
        self.tag_structure = {'Employment': {
                                        'StatusEmploy': ['employed', 'unemployed', 'retired', 'on_disability', 'student', 'homemaker', ''],
                                        'Duration': ['NA'],
                                        'History': ['NA'],
                                        'Type': ['NA']},
                              'LivingStatus': {
                                        'StatusTime': ['current', 'past', 'future', 'none', ''],
                                        'TypeLiving': ['alone', 'with_family', 'with_others', 'homeless', ''],
                                        'Duration': ['NA'],
                                        'History': ['NA']},
                              'Alcohol': {
                                        'StatusTime': ['none', 'current', 'past', 'future', ''],
                                        'Duration': ['NA'],
                                        'History': ['NA'],
                                        'Method': ['NA'],
                                        'Type': ['NA'],
                                        'Amount': ['NA'],
                                        'Frequency': ['NA']},
                              'Drug': {
                                        'StatusTime': ['none', 'current', 'past', 'future', ''],
                                        'Duration': ['NA'],
                                        'History': ['NA'],
                                        'Method': ['NA'],
                                        'Type': ['NA'],
                                        'Amount': ['NA'],
                                        'Frequency': ['NA']},
                              'Tobacco': {
                                        'StatusTime': ['none', 'current', 'past', 'future', ''],
                                        'Duration': ['NA'],
                                        'History': ['NA'],
                                        'Method': ['NA'],
                                        'Type': ['NA'],
                                        'Amount': ['NA'],
                                        'Frequency': ['NA']}
                              }
        self.bio_trigger_tags = ['O'] + ['B-' + t for t in self.tag_structure] + ['I-' + t for t in self.tag_structure]
        self.bio_trigger_tag_to_index = {tag: index for index, tag in enumerate(self.bio_trigger_tags)}
        self.bio_tags = {e: {
            task: ['O'] + ['B-' + task + '-' + t for t in self.tag_structure[e][task]] + ['I-' + task + '-' + t for t in
                                                                                          self.tag_structure[e][task]]
            for task in self.tag_structure[e]} for e in self.tag_structure}
        self.bio_tags_to_index = {
            e: {task: {tag: index for index, tag in enumerate(seq)} for task, seq in self.bio_tags[e].items()} for e in
            self.bio_tags}
        self.eval()

    def save_model(self, path=None):
        if not path:
            path=self.model_dir + '/model.p'
        print ('saving model', path)
        init_time = time.time()
        with open(path, 'wb') as f:
           pickle.dump(self, f)#, pickle.HIGHEST_PROTOCOL)
        print('saved t:', round(time.time() - init_time, 2), 's')

    def add_embeddings(self, path_to_embedding_file):
        embs = KeyedVectors.load_word2vec_format(path_to_embedding_file)

        print(embs)
        if not self.pretrained_embs:
            # reset vocab to that of embeddings
            self.vocab = embs.key_to_index
            self.word_emb_dim=embs.vectors[0].shape[0]
            # set embeddings with loaded weights
            self.word_embs = nn.Embedding(num_embeddings=len(self.vocab), embedding_dim=embs.vectors[0].shape[0])
            self.word_embs.weight=nn.Parameter(torch.tensor(embs.vectors,dtype=torch.float32))
            # prevent training/updating the embeddings
            self.word_embs.weight.requires_grad = False
            self.pretrained_embs=1

        elif self.pretrained_embs:
            print('current emb dims',self.word_embs.weight.shape)
            all_words = set(self.vocab).union(set(embs.key_to_index))
            newembdim = self.word_emb_dim + embs.vectors[0].shape[0]
            newweights = torch.zeros((len(all_words),newembdim))
            new_vocab = {w:i for i,w in enumerate(all_words)}

            for w,ix in new_vocab.items():
                wemb1 = self.word_embs.weight[self.vocab[w]] if w in self.vocab else self.word_embs.weight[self.vocab[self.unk]]
                wemb2 = torch.tensor(embs[w] if w in embs else embs[self.unk])
                newemb = torch.cat([wemb1,wemb2],dim=0)
                newweights[ix]=newemb

            newembs = nn.Embedding(num_embeddings=len(new_vocab), embedding_dim=newembdim)
            newembs.weight = nn.Parameter(torch.tensor(newweights.detach(),dtype=torch.float32)) # TODO: check float 64? what is the default?
            self.word_embs = newembs
            self.word_emb_dim = newembdim
            self.word_embs.weight.requires_grad = False
            self.vocab=new_vocab
            print('new embs dims',self.word_embs.weight.shape)

        return self
        # TODO: ensure embeddings are not updated during training.

    def preproc_token(self, string):
        if self.conflate_digits:
            string = re.sub('\d', '5', string)
        if self.lowercase:
            string = string.lower()

        # during training sometimes replace a rare word (of freq 1) with unk, to train the unk embedding (Good–Turing frequency estimation).
        if self.training and string in self.wordsoffreq1 and random.random() < self.unk1_prob:
            string = self.unk
        return string

    def get_gt_trigger_tags_from_doc(self, doc, tag_indices=False):
        gt_trigger_tags = ['O' for _ in sum(doc.tokens, [])]
        subtask_trigger_tags = {}
        # make GT trigger tag sequence
        for e in doc.events():
            # event trigger
            especification = (e.type_, e.arguments[0].token_start, e.arguments[0].token_end)
            gt_trigger_tags[e.arguments[0].token_start] = 'B-' + e.type_
            for i in range(e.arguments[0].token_start + 1, e.arguments[0].token_end):
                gt_trigger_tags[i] = 'I-' + e.type_

            # subtasks
            subtask_trigger_tags[especification] = {subtask: ['O' for _ in sum(doc.tokens, [])] for subtask in
                                                    self.bio_tags[e.type_]}
            for arg in e.arguments[1:]:
                if not arg.type_ in self.bio_tags[e.type_]:
                    #print('ignored annotation:', doc.id, arg.type_) # TODO: consider annotations not in task specification?
                    continue
                subtask_trigger_tags[especification][arg.type_][arg.token_start] = 'B-' + arg.type_ + '-' + (
                    arg.subtype if arg.subtype else 'NA')
                for i in range(arg.token_start + 1, arg.token_end):
                    subtask_trigger_tags[especification][arg.type_][i] = 'I-' + arg.type_ + '-' + (
                        arg.subtype if arg.subtype else 'NA')
                # print('subtask_gt_trigger_tags',arg.type_,subtask_trigger_tags[especification][arg.type_])
            if tag_indices:
                for subtask in self.bio_tags[e.type_]:
                    if tag_indices:
                        subtask_trigger_tags[especification][subtask] = [self.bio_tags_to_index[e.type_][subtask][tag]
                                                                         for tag in
                                                                         subtask_trigger_tags[especification][subtask]]

        if tag_indices:
            gt_trigger_tags = [self.bio_trigger_tag_to_index[t] for t in gt_trigger_tags]

        return gt_trigger_tags, subtask_trigger_tags

    def build(self):
        # 1 word embedding layer
        if not self.pretrained_embs:
            self.word_embs = nn.Embedding(len(self.vocab), self.word_emb_dim)

        # 2 biRNN (LSTM or GRU)
        if self.rnn_type.lower() == 'gru':
            print('using GRUs')
            self.rnn = nn.GRU(self.word_emb_dim, self.rnn_dim, num_layers=self.rnn_num_layers, bidirectional=True)
        else:
            print('using LSTMs')
            self.rnn = nn.LSTM(self.word_emb_dim, self.rnn_dim, num_layers=self.rnn_num_layers, bidirectional=True)

        self.dropout_layer = nn.Dropout(self.dropout)
        self.emb_dropout_layer = nn.Dropout(self.emb_dropout)

        # 3.1 Event triggers layer
        self.trigger_layer = nn.Linear(in_features=self.rnn_dim * 2, out_features=len(self.bio_trigger_tags))

        # 3.2 task-specific layers


        for trigger_type in self.tag_structure:
            self.subtask_layers[trigger_type] = nn.ModuleDict()
            if self.divide_subtask_networks:
                if self.pretrained_embs:
                    self.subtask_wembs[trigger_type] = self.word_embs
                else:
                    self.subtask_wembs[trigger_type] = nn.Embedding(len(self.vocab), self.word_emb_dim)

                self.subtask_rnns[trigger_type]= nn.GRU(self.word_emb_dim, self.rnn_dim, num_layers=self.rnn_num_layers, bidirectional=True) if self.rnn_type.lower() == 'gru' else nn.LSTM(self.word_emb_dim, self.rnn_dim, num_layers=self.rnn_num_layers, bidirectional=True)

            for subtask in self.tag_structure[trigger_type]:
                self.subtask_layers[trigger_type][subtask] = nn.Linear(in_features=1 + self.rnn_dim * 2,
                                                                       out_features=len(
                                                                           self.bio_tags[trigger_type][subtask]))

    def construct_vocabulary(self, traindocs):
        tokenlist = [self.preproc_token(t) for d in traindocs.values() for ts in d.tokens for t in ts]
        wordfreqs = Counter(tokenlist)
        self.wordsoffreq1 = set([w for w,c in wordfreqs.items() if c==1])
        vocab = dict([(v, i) for (i, v) in enumerate(sorted(list(wordfreqs.keys())))])
        vocab[self.unk] = len(vocab)
        return vocab

    def selftrain_fit(self, selftrainpath, trainpath, trainpath2=False, lossweight2=1, devpath=False, max_num_epochs=100, lr=0.001, grad_clip=5):

        # 1. train the model as ususual
        print('Training initial model')
        self.fit(trainpath,trainpath2,lossweight2,devpath,max_num_epochs,lr,grad_clip)

        # 2. make predictions in selftraining data
        print('Making predictions in self training data')

        tmpdir = "./tmpselftrainingpreds-"+str(time.time())
        self.predict_all_txts_in_directory(selftrainpath, tmpdir)

        # 3. add current training data to the temporary self training data
        trainfiles = os.listdir(trainpath)
        for filepath in trainfiles:
            filepath = trainpath +'/' +filepath
            #print('copy',filepath,tmpdir)
            shutil.copy(filepath,tmpdir)

        # 4. refit the model on all data
        print('Training the final model (incl. selftraining data)')
        self.fit(tmpdir,trainpath2,lossweight2,devpath,max_num_epochs,lr,grad_clip)

        # 5. remove the temporary directory
        shutil.rmtree(tmpdir)

    def freeze_all_layers_but_the_bias_terms(self):
        for param in self.parameters():
            param.requires_grad = False

        self.trigger_layer.bias.requires_grad = True
        self.trigger_layer.weight.requires_grad = True

        for trigger_type in self.tag_structure:
            for subtask in self.tag_structure[trigger_type]:
                self.subtask_layers[trigger_type][subtask].bias.requires_grad=True
                self.subtask_layers[trigger_type][subtask].weight.requires_grad=True


    def calibrate(self, trainpath, max_num_epochs=200, lr=0.001):
        self.freeze_all_layers_but_the_bias_terms()
        self.fit(trainpath, max_num_epochs=max_num_epochs if max_num_epochs > 100 else 100, lr=lr, calibration_only=True, grad_clip=5)

    def fit(self, trainpath, trainpath2=False, lossweight2=1, devpath=False, max_num_epochs=100, lr=0.001, grad_clip=5, calibration_only=0):
        traincorpus = Corpus()
        traincorpus.import_dir(path=trainpath)
        traindocs = traincorpus.docs(as_dict=True)

        if trainpath2:
            train2corpus = Corpus()
            train2corpus.import_dir(path=trainpath2)
            traindocs2 = train2corpus.docs(as_dict=True)

            # add them to training
            for k,v in traindocs2.items():
                traindocs[k]=v

        if not calibration_only:
            if not self.pretrained_embs:
                self.vocab = self.construct_vocabulary(traindocs)
            self.build()

        if devpath:
            devcorpus = Corpus()
            devcorpus.import_dir(path=devpath)
            devdocs = devcorpus.docs(as_dict=True)

        self.optimizer = torch.optim.Adam(model.parameters(),lr=lr, amsgrad=True)
        #if calibration_only:
        #    self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        if self.pretrained_embs:
            self.word_embs.weight.requires_grad = False
        if calibration_only:
            self.freeze_all_layers_but_the_bias_terms()

        document_keys = list(traindocs.keys())
        epoch_times = []
        self.train()
        print('No. training documents used:', len(traindocs))
        for epoch in range(1,max_num_epochs+1):
            t0 = time.time()
            random.shuffle(document_keys)
            eloss = []
            for d_i in document_keys:
                self.optimizer.zero_grad()
                tr, st = self.forward(traindocs[d_i])
                loss = self.loss(traindocs[d_i], tr, st)
                if trainpath2 and lossweight2 and d_i in traindocs2:
                    loss = loss * lossweight2
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                self.optimizer.step()
                eloss += [loss.detach().numpy()]


            if devpath:
                self.eval()
                self.predict_all_txts_in_directory(devpath, output_dir='./devpreds')
                predcorp = Corpus()
                predcorp.import_dir('./devpreds')
                deveval = score_docs(devdocs, predcorp.docs(as_dict=True), \
                           labeled_args=SDOH_LABELED_ARGUMENTS,
                           score_trig=OVERLAP,
                           score_span=EXACT,
                           score_labeled=LABEL,
                           output_path=None,
                           include_detailed=False)
                print(deveval.loc[deveval['argument'].isin(['Trigger','OVERALL'])][['event','F1']].set_index('event'))
                self.train()

            epoch_time = time.time()-t0
            epoch_times.append(epoch_time)
            #print(self.trigger_layer.weight[0])
            #print(self.trigger_layer.bias)

            print('Epoch',epoch,'\tLoss:', numpy.mean(eloss),'\tTook:',round(epoch_time,1),'s','Time left:', round(numpy.mean(epoch_times)*(max_num_epochs-epoch),1), 's', 'ETA:', (datetime.datetime.now() + datetime.timedelta(seconds=numpy.mean(epoch_times)*(max_num_epochs-epoch))).strftime('%H:%M on %d %b %Y '))
        self.eval()

    def forward(self, doc, gt_triggers=True):
        # 1.1 encode text
        token_seq = sum(doc.tokens, [])
        # print(token_seq)
        token_index_seq = torch.tensor([self.vocab[self.preproc_token(t)] if t in self.vocab else self.vocab[self.unk] for t in token_seq])
        # print('token_index_seq',token_index_seq.shape)
        embedded_seq = self.word_embs(token_index_seq)
        if self.emb_dropout:
            embedded_seq = self.emb_dropout_layer(self.word_embs(token_index_seq))

        # print('embedded_seq',embedded_seq.shape)
        hidden_states, _ = self.rnn(embedded_seq.view(len(token_seq), 1, self.word_emb_dim))
        dropped_out_hidden_states = self.dropout_layer(hidden_states)

        # 2.1 get trigger scores
        if self.relus:
            raw_trigger_scores = torch.nn.functional.leaky_relu(self.trigger_layer(dropped_out_hidden_states))
        else:
            raw_trigger_scores = self.trigger_layer(dropped_out_hidden_states)

        trigger_scores = nn.functional.log_softmax(raw_trigger_scores, 2)  # output

        # 3.1 get labeled argument candidates
        if gt_triggers:
            trigger_tags, _ = self.get_gt_trigger_tags_from_doc(doc)
        else:
            trigger_tags = [self.bio_trigger_tags[i] for i in torch.argmax(trigger_scores, 2)]

        events = get_event_trigger_info_from_bio_tags(trigger_tags)

        # 3.2 predict labeled argument probs
        event_labels_and_arguments_scores = []
        for (etype, start, end) in events:
            # print(etype, start, end)
            # dist_vec = torch.tensor([max([start-i, 0, i-end]) for i in range(0,len(token_seq))]).view(len(token_seq),1,1)
            indicator_vec = torch.tensor([1 if i in range(start, end) else 0 for i in range(0, len(token_seq))]).view(len(token_seq), 1, 1)
            if self.divide_subtask_networks:
                if not self.pretrained_embs:
                    embedded_seq = self.subtask_wembs[etype](token_index_seq)
                if self.emb_dropout:
                    embedded_seq = self.emb_dropout_layer(self.word_embs(token_index_seq))

                hidden_states, _ = self.subtask_rnns[etype](embedded_seq.view(len(token_seq), 1, self.word_emb_dim))
                dropped_out_hidden_states = self.dropout_layer(hidden_states)
            input_vector_to_subtask_classifier = torch.cat([dropped_out_hidden_states, indicator_vec], 2)

            subtask_scores = {}
            for subtask in self.tag_structure[etype]:
                # print('subtask',subtask)
                if self.relus:
                    raw_subtask_trigger_scores = torch.nn.functional.leaky_relu(self.subtask_layers[etype][subtask](input_vector_to_subtask_classifier))
                else:
                    raw_subtask_trigger_scores = self.subtask_layers[etype][subtask](input_vector_to_subtask_classifier)

                subtask_scores[subtask] = nn.functional.log_softmax(raw_subtask_trigger_scores, 2)
                # print('subtask_scores[subtask]',subtask_scores[subtask].shape)
            event_labels_and_arguments_scores += [((etype, start, end), subtask_scores)]

        return trigger_scores, event_labels_and_arguments_scores

    def loss(self, doc, trigger_scores, event_labels_and_arguments_scores):
        losses = {}

        criterion = nn.NLLLoss()
        gt_trigger_tags, gt_subtask_tags = self.get_gt_trigger_tags_from_doc(doc, tag_indices=True)
        losses['triggers'] = criterion(trigger_scores.view(len(sum(doc.tokens, [])), len(self.bio_trigger_tags)),
                                       torch.tensor(gt_trigger_tags)).mean()

        # print('gt_subtask_tags',gt_subtask_tags)
        losses['subtasks'] = 0
        for (espec, subtask_scores) in event_labels_and_arguments_scores:
            for subtask in subtask_scores:
                if espec in gt_subtask_tags:
                    subtask_in = subtask_scores[subtask].view(len(sum(doc.tokens, [])),
                                                              len(self.bio_tags[espec[0]][subtask]))
                    losses['subtasks'] += criterion(subtask_in, torch.tensor(gt_subtask_tags[espec][subtask]))
                    # print(espec, gt_subtask_tags[espec])
                else:
                    print('missing espec', espec)

        return losses['triggers'] + self.subtask_weight * losses['subtasks']

    def predict_all_txts_in_directory(self, input_dir, output_dir='predictions'):

        for file in os.listdir(input_dir):
            if '.txt' in file:
                fname = file.rstrip('.txt')
                if not os.path.exists(os.path.join(input_dir, fname+'.ann')):
                    with open(os.path.join(input_dir, fname+'.ann'), 'w') as f:
                        f.write('')

        testtxtcorpus = Corpus()
        testtxtcorpus.import_dir(path=input_dir)

        if output_dir:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir)

        for doc_id, doc in testtxtcorpus.docs(as_dict=True).items():
            predicted_annotations = self.predict(doc)
            ann_out_path = output_dir + '/' + doc_id + '.ann'
            with open(ann_out_path, 'w') as f:
                f.write(predicted_annotations)
            # copy text
            txt_out_path = output_dir + '/' + doc_id + '.txt'
            with open(txt_out_path, 'w') as f:
                f.write(doc.text)

    def predict(self, doc):
        #print('PRED', doc.id)
        offset_list = sum(doc.token_offsets, [])
        doctext = doc.text.replace('\n', ' ')

        # 1 forward on text docs
        trigger_scores, event_labels_and_arguments_scores = self.forward(doc, gt_triggers=False)
        trigger_tags = [self.bio_trigger_tags[i] for i in torch.argmax(trigger_scores, 2)]

        Ts, Es, As = [], [], []

        #print(trigger_tags)

        for ((etype, start, end), subtask_scores) in event_labels_and_arguments_scores:
            #print('>', etype, start, end)
            T_str = "T" + str(len(Ts) + 1) + '\t' + etype + ' ' + str(offset_list[start][0]) + ' ' + str(
                offset_list[end - 1][-1]) + '\t' + doctext[offset_list[start][0]:offset_list[end - 1][-1]]
            Ts.append(T_str)

            arg_summaries = [etype + ":T" + str(len(Ts))]
            for subtask in subtask_scores:
                # print(subtask_scores[subtask])
                arguments = get_event_trigger_info_from_bio_tags(
                    [self.bio_tags[etype][subtask][i] for i in torch.argmax(subtask_scores[subtask], 2)])
                #print(arguments)
                if len(arguments) > 0:
                    a_type, a_start, a_end = arguments[0]
                    T_str = "T" + str(len(Ts) + 1) + '\t' + a_type.split('-')[0] + ' ' + str(
                        offset_list[a_start][0]) + ' ' + str(offset_list[a_end - 1][-1]) + '\t' + doctext[
                                                                                                  offset_list[a_start][
                                                                                                      0]:offset_list[
                                                                                                      a_end - 1][-1]]
                    TE_str = a_type.split('-')[0] + ":T" + str(len(Ts) + 1)
                    Ts.append(T_str)
                    arg_summaries.append(TE_str)
                    if a_type.split('-')[1] != 'NA' and a_type.split('-')[1] != '':
                        #print('ARG', a_type.split('-')[1])
                        A_str = "A" + str(len(As) + 1) + '\t' + a_type.split('-')[0] + 'Val T' + str(len(Ts)) + ' ' + \
                                a_type.split('-')[1]
                        As.append(A_str)
            E_str = "E" + str(len(Es) + 1) + '\t' + ' '.join(arg_summaries)
            Es.append(E_str)

        ann = '\n'.join(Ts + Es + As)

        return ann


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Script to train and test a SDOH model for the N2C2 Shared Task (2022).')
    parser.add_argument('-labeled_train_corpus', required=False, help='labeled_train_corpus_path', default=None)
    parser.add_argument('-second_labeled_train_corpus', required=False, help='a second labeled_train_corpus_path (target)', default=None)
    parser.add_argument('-labeled_dev_corpus', required=False, help='labeled_dev_corpus_path', default=None)
    parser.add_argument('-output_test_predictions', required=False, help='predictions_test_corpus_path', default="./predictions/")
    parser.add_argument('-unlabeled_test_corpus', required=False, help='unlabeled_test_corpus_path', default=None)
    parser.add_argument('-labeled_test_corpus', required=False, help='unlabeled_test_corpus_path', default=None)
    parser.add_argument('-model_saving_path', required=False, help='model_saving_path', default=None)
    parser.add_argument('-model_loading_path', required=False, help='model_loading_path', default=None)
    parser.add_argument('-max_num_epochs', required=False, help='max_num_epochs (default: 250)', type=int, default=250)
    parser.add_argument('-conflate_digits', required=False, help='conflate digits (flag: 1 or 0, default: 0)', type=int, default=0)
    parser.add_argument('-lowercase_tokens', required=False, help='lowercase tokens (flag: 1 or 0, default: 0)', type=int, default=0)
    parser.add_argument('-unk1_prob', required=False, help='replace 1 time occuring tokens sometimes with unk to train the unk embedding, default prob: 0.5', type=float, default=0.5)
    parser.add_argument('-wemb_size', required=False, help='word embedding size, default: 200', type=int, default=200)
    parser.add_argument('-rnn_dim', required=False, help='LSTM/GRU dimension, default: 50', type=int, default=100)
    parser.add_argument('-rnn_type', required=False, help='LSTM or GRU, default: GRU', type=str, default='GRU')
    parser.add_argument('-output_eval', required=False, help='Output evaluation file', type=str, default="evaluation.csv")
    parser.add_argument('-dropout', required=False, help='Dropout probability, default: 0.2', type=float, default=0.2)
    parser.add_argument('-lr', required=False, help='Learning rate, default: 0.005', type=float, default=0.005)
    parser.add_argument('-relus', required=False, help='Use Relu activations, default: 1', type=int, default=1)
    parser.add_argument('-emb_dropout', required=False, help='Use dropout on embedding layer, default: 0', type=float, default=0)
    parser.add_argument('-subtask_weight', required=False, help='Loss weight assigned to the subtask loss (0, inf), default: 1', type=float, default=1)
    parser.add_argument('-clip_gradient', required=False, help='Clip gradients during training, default: 5', type=float, default=5)
    parser.add_argument('-rnn_depth', required=False, help='Number of sequential RNN layers, default: 1', type=int, default=1)
    parser.add_argument('-divide_subtask_networks', required=False, help='Use separate RNNs for each subtask, default: 0', type=float, default=0)
    parser.add_argument('-pretrained_wembs1', required=False, help='Use pretrained embeddings set 1, provide path to embeddding .txt or .vec file, default: 0', type=str, default=0)
    parser.add_argument('-pretrained_wembs2', required=False, help='Use pretrained embeddings set 2, provide path to embeddding .txt or .vec, default: 0', type=str, default=0)
    parser.add_argument('-loss_weight_train2', required=False, help='Weight for the loss for documents in training set 2 (if provided), default: 1', type=float, default=1)
    parser.add_argument('-selftraining', required=False, help='Apply self training to a provided (unlabeled) dataset of .txt files, default: 0', type=str, default=0)
    parser.add_argument('-calibrate', required=False, help='Calibrate the model on the target files, default: 0', type=str, default=0)
    args = parser.parse_args()

    # 1.1 Initialize a model
    model = sdoh_model(conflate_digits=args.conflate_digits, lowercase=args.lowercase_tokens, wemb_size=args.wemb_size,
                       rnn_dim=args.rnn_dim, subtask_weight=args.subtask_weight, dropout=args.dropout, relus=args.relus,
                       emb_dropout=args.emb_dropout, rnn_type=args.rnn_type,
                       divide_subtask_networks=args.divide_subtask_networks, rnn_num_layers=args.rnn_depth)

    # 1.2 Load or train
    if args.model_loading_path:
        print('Loading model from',args.model_loading_path)
        #newdict = torch.load(args.model_loading_path)
        #newdict = {key.replace("module.", ""): value for key, value in newdict.items()}
        #model.load_state_dict(newdict)
        #model.load_state_dict(torch.load(args.model_loading_path), strict=False)
        model = load_model(args.model_loading_path)
        model.eval()
    else:
        # Use preptrained embeddings?
        if args.pretrained_wembs1:
            print('Loading embeddings from',args.pretrained_wembs1)
            model = model.add_embeddings(args.pretrained_wembs1)
        if args.pretrained_wembs2:
            print('Adding embeddings from',args.pretrained_wembs2)
            model = model.add_embeddings(args.pretrained_wembs2)

        # Training
        if args.selftraining:
            model.selftrain_fit(selftrainpath=args.selftraining, trainpath=args.labeled_train_corpus,
                                trainpath2=args.second_labeled_train_corpus,
                      lossweight2=args.loss_weight_train2, devpath=args.labeled_dev_corpus,
                      max_num_epochs=args.max_num_epochs, lr=args.lr, grad_clip=args.clip_gradient)
        else:
            model.fit(trainpath=args.labeled_train_corpus, trainpath2=args.second_labeled_train_corpus,
                      lossweight2=args.loss_weight_train2, devpath=args.labeled_dev_corpus,
                      max_num_epochs=args.max_num_epochs, lr=args.lr, grad_clip=args.clip_gradient)

        if args.calibrate: # only refit bias terms in a given dataset
            model.calibrate(args.calibrate, max_num_epochs=args.max_num_epochs, lr=args.lr)


        if args.model_saving_path:
            print('Saving trained model to',args.model_saving_path)
            model.save_model(args.model_saving_path)
            #torch.save(model.state_dict(), args.model_saving_path)

    if args.unlabeled_test_corpus and args.output_test_predictions:
        # 2. Make predictions in the test set
        model.predict_all_txts_in_directory(args.unlabeled_test_corpus, args.output_test_predictions)
    else:
        print("No predictions made: either no unlabeled_test_corpus or output_test_predictions provided.")

    if args.labeled_test_corpus:
        print('Evaluating in', args.labeled_test_corpus)
        #3. Evaluate the predictions
        evaluation = score_brat_sdoh( \
            gold_dir=args.labeled_test_corpus,
            predict_dir=args.output_test_predictions,
            labeled_args=SDOH_LABELED_ARGUMENTS,
            score_trig=OVERLAP,
            score_span=EXACT,
            score_labeled=LABEL,
            output_path=args.output_eval,
            include_detailed=False,
            loglevel='info')
        print('Evaluation saved to', args.output_eval)

        micro_average_subtypes_csv(args.output_eval, '.'.join(args.output_eval.split('.')[:-1]+['_microaveraged']+['.csv']))
        pandas.set_option("display.max_rows", 100, "display.max_columns", 100)
        pandas.options.display.width = 0
        print(evaluation)

