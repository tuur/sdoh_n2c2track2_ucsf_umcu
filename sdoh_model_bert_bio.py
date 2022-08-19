import random, os, shutil, numpy, torch, pandas, time, datetime, re, argparse
from collections import Counter
import numpy as np
from torch import nn
from transformers import AutoTokenizer, AutoModel

from brat_scoring.corpus import Corpus
from brat_scoring.constants import EXACT, LABEL, OVERLAP
from brat_scoring.scoring import score_brat_sdoh, score_docs, micro_average_subtypes_csv
from brat_scoring.constants_sdoh import LABELED_ARGUMENTS as SDOH_LABELED_ARGUMENTS

random.seed(1)


def get_event_trigger_info_from_bio_tags(bio_tags):
    events = []
    inevent = False
    start = 0
    etype = ''
    for i, tag in enumerate(bio_tags):
        if tag == 'SKIP':
            continue
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


class sdoh_bert_model(nn.Module):
    def __init__(self, bert_model_name_or_path, is_roberta=False,
                 max_seq_len=512, freeze_emb=True, use_attn=True,
                 conflate_digits=True, lowercase=True, unk1_prob=0.5, wemb_size=768, rnn_dim=50,
                 rnn_type='LSTM', subtask_weight=1, dropout=0.5, relus=1, emb_dropout=0, divide_subtask_networks=0,
                 rnn_num_layers=1):
        super().__init__()
        self.bert_model_name_or_path = bert_model_name_or_path
        self.is_roberta = is_roberta
        self.freeze_embeddings = freeze_emb
        self.use_attn = use_attn
        self.max_seq_len=max_seq_len
        self.word_emb_dim = wemb_size
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
        self.bio_trigger_tags = ['O', 'SKIP'] + ['B-' + t for t in self.tag_structure] + ['I-' + t for t in self.tag_structure]
        self.bio_trigger_tag_to_index = {tag: index for index, tag in enumerate(self.bio_trigger_tags)}
        self.bio_trigger_tag_to_index['SKIP'] = -100
        self.bio_tags = {e: {
            task: ['O', 'SKIP'] + ['B-' + task + '-' + t for t in self.tag_structure[e][task]] + ['I-' + task + '-' + t for t in
                                                                                          self.tag_structure[e][task]]
            for task in self.tag_structure[e]} for e in self.tag_structure}
        self.bio_tags_to_index = {
            e: {task: {tag: (index if tag != 'SKIP' else -100) for index, tag in enumerate(seq)} for task, seq in self.bio_tags[e].items()} for e in
            self.bio_tags}
        self.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preproc_token(self, string):
        if self.conflate_digits:
            string = re.sub('\d', '5', string)
        if self.lowercase:
            string = string.lower()

        # during training sometimes replace a rare word (of freq 1) with unk, to train the unk embedding (Good–Turing frequency estimation).
        if self.training and string in self.wordsoffreq1 and random.random() < self.unk1_prob:
            string = self.unk
        return string

    def _get_bert_mapping(self, encodings, tags, tag2id, tag_indices):
        # This function only works for 1D tags, i.e., tags from a single document.
        # For 2D, refer to: https://huggingface.co/transformers/v3.2.0/custom_datasets.html#ft-native
        mismatch = False
        if tag_indices:
            labels = tags
        else:
            labels = [tag2id[tag] for tag in tags]

        id2tag = {i:k for k, i in tag2id.items()}
        # id2tag[-100] = 'SKIP'

        # squeeze() is necessary since our batch size is 1. Update the code accordingly if using > 1 batch size
        doc_enc_labels = np.ones(len(encodings.offset_mapping.squeeze().tolist()), dtype=int) * -100
        arr_offset = encodings.offset_mapping.squeeze().numpy()

        # set labels whose first offset position is 0 and the second is not 0
        num_starts = (
                (arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)
            ).sum()

        if num_starts < len(labels):
            doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = labels[:num_starts]
            mismatch = True
        else:
            doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = labels

        encoded_labels = doc_enc_labels.tolist()

        if not tag_indices:
            encoded_labels = [id2tag[label] for label in encoded_labels]

        return encoded_labels, mismatch

    def get_gt_trigger_tags_from_doc(self, doc, tag_indices=False, bert_feats=None, bert_mapping=True, doc_id=None):
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
                    # print('ignored annotation:', doc.id, arg.type_) # TODO: consider annotations not in task specification?
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

        if bert_mapping:

            gt_trigger_tags, mismatch = self._get_bert_mapping(bert_feats, gt_trigger_tags, self.bio_trigger_tag_to_index,
                                                     tag_indices)

            if mismatch:
                print("Potentially truncated doc. Could also be encoding error; needs to be manually verified.")
                if doc_id:
                    print("Doc ID: ", doc_id)
                else:
                    print("Doc: ", doc)
            # Iterate over dictionary and pass as list:
            for (etype, start, end), subtypes in subtask_trigger_tags.copy().items():
                for subtype, subtype_tags in subtypes.items():
                    subtask_trigger_tags[(etype, start, end)][subtype], mismatch = self._get_bert_mapping(
                        bert_feats, subtype_tags,
                        self.bio_tags_to_index[etype][subtype],
                        tag_indices)

        return gt_trigger_tags, subtask_trigger_tags

    def build(self):
        # 1 word embedding layer
        print("BERT model name or path: ", self.bert_model_name_or_path)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name_or_path,
                                                           add_prefix_space=self.is_roberta)
        self.word_embs = AutoModel.from_pretrained(self.bert_model_name_or_path)
        if self.freeze_embeddings:
            for param in self.word_embs.parameters():
                param.requires_grad = False

        # 2 biRNN (LSTM or GRU)
        if self.rnn_type.lower() == 'gru':
            print('using GRUs')
            self.rnn = nn.GRU(self.word_emb_dim, self.rnn_dim, num_layers=self.rnn_num_layers, bidirectional=True)
        else:
            print('using LSTMs')
            self.rnn = nn.LSTM(self.word_emb_dim, self.rnn_dim, num_layers=self.rnn_num_layers, bidirectional=True)

        self.attn_layer = nn.Linear(in_features=2*self.rnn_dim, out_features=1, bias=False)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.emb_dropout_layer = nn.Dropout(self.emb_dropout)

        # 3.1 Event triggers layer
        self.trigger_layer = nn.Linear(in_features=self.rnn_dim * 2, out_features=len(self.bio_trigger_tags))

        # 3.2 task-specific layers
        self.subtask_layers = nn.ModuleDict()
        self.subtask_rnns = nn.ModuleDict()

        for trigger_type in self.tag_structure:
            self.subtask_layers[trigger_type] = nn.ModuleDict()
            if self.divide_subtask_networks:
                self.subtask_rnns[trigger_type] = nn.GRU(self.word_emb_dim, self.rnn_dim, num_layers=self.rnn_num_layers,
                                                        bidirectional=True) if self.rnn_type.lower() == 'gru' else nn.LSTM(self.word_emb_dim, self.rnn_dim, num_layers=self.rnn_num_layers, bidirectional=True)

            self.subtask_attn_layer = nn.Linear(in_features=2 * self.rnn_dim, out_features=1, bias=False)
            for subtask in self.tag_structure[trigger_type]:
                self.subtask_layers[trigger_type][subtask] = nn.Linear(in_features=1 + self.rnn_dim * 2,
                                                                       out_features=len(
                                                                           self.bio_tags[trigger_type][subtask]))

        self.to(self.device)

    def construct_vocabulary(self, traindocs):
        tokenlist = [self.preproc_token(t) for d in traindocs.values() for ts in d.tokens for t in ts]
        wordfreqs = Counter(tokenlist)
        self.wordsoffreq1 = set([w for w,c in wordfreqs.items() if c==1])
        vocab = dict([(v, i) for (i, v) in enumerate(sorted(list(wordfreqs.keys())))])
        vocab[self.unk] = len(vocab)
        return vocab

    def selftrain_fit(self, selftrainpath, trainpath, trainpath2=False, lossweight2=1, devpath=False,
                      max_num_epochs=100, lr=0.001, grad_clip=5):

        # 1. train the model as ususual
        print('Training initial model')
        self.fit(trainpath,trainpath2,lossweight2,devpath,max_num_epochs,lr,grad_clip)

        # 2. make predictions in selftraining data
        print('Making predictions in self training data')

        tmpdir = "./tmpselftrainingpreds-"+str(time.time())

        selftrain_corpus = Corpus()
        selftrain_corpus.import_dir(path=selftrainpath)
        selftrain_docs = selftrain_corpus.docs(as_dict=True)
        selftrain_tokens = dict()
        for doc_id, doc in selftrain_docs.items():
            tokens = [tok for sent in doc.tokens for tok in sent]
            selftrain_tokens[doc_id] = self.bert_tokenizer(tokens, is_split_into_words=True,
                                                           return_offsets_mapping=True,
                                                           padding=True,
                                                           truncation=True,
                                                           max_length=self.max_seq_len,
                                                           add_special_tokens=False,
                                                           return_tensors='pt')
        self.predict_all_txts_in_directory(selftrain_docs, selftrain_tokens, output_dir=tmpdir)

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
        self.fit(trainpath, max_num_epochs=max_num_epochs if max_num_epochs > 100 else 100, lr=lr,
                 calibration_only=True, grad_clip=5)

    def fit(self, trainpath, trainpath2=False, lossweight2=1, devpath=False, max_num_epochs=100, lr=0.001, grad_clip=5,
            calibration_only=0):

        if not calibration_only:
            self.build()

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

        train_tokens = dict()

        for doc_id, doc in traindocs.items():
            tokens = [tok for sent in doc.tokens for tok in sent]
            train_tokens[doc_id] = self.bert_tokenizer(tokens, is_split_into_words=True,
                                                       return_offsets_mapping=True,
                                                       padding=True,
                                                       truncation=True,
                                                       max_length=self.max_seq_len,
                                                       add_special_tokens=False,
                                                       return_tensors='pt'
                                                       )

        if not calibration_only:
            self.vocab = self.construct_vocabulary(traindocs)

        assert len(traindocs.keys()) == len(train_tokens.keys()), "Mismatch in number of documents, tokens"

        if devpath:
            devcorpus = Corpus()
            devcorpus.import_dir(path=devpath)
            devdocs = devcorpus.docs(as_dict=True)
            dev_tokens = dict()
            for doc_id, doc in devdocs.items():
                tokens = [tok for sent in doc.tokens for tok in sent]
                dev_tokens[doc_id] = self.bert_tokenizer(tokens, is_split_into_words=True,
                                                           return_offsets_mapping=True,
                                                           padding=True,
                                                           truncation=True,
                                                           max_length=self.max_seq_len,
                                                           add_special_tokens=False,
                                                           return_tensors='pt')


        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=lr, amsgrad=True)
        document_keys = list(traindocs.keys())
        epoch_times = []
        self.train()
        self.to(self.device)

        print("Starting training")
        epoch = 1
        while epoch < max_num_epochs+1:
            t0 = time.time()
            random.shuffle(document_keys)
            eloss = []
            for d_i in document_keys:
                self.optimizer.zero_grad()
                try:
                    tr, st = self.forward(traindocs[d_i], train_tokens[d_i], doc_id=d_i)
                except Exception as e:
                    print("Exception: ", e)
                    print("Fails in train doc: ", d_i, "tokens:", train_tokens[d_i])
                    continue
                loss = self.loss(traindocs[d_i], train_tokens[d_i], tr, st, doc_id=d_i)
                if trainpath2 and lossweight2 and d_i in traindocs2:
                    loss = loss * lossweight2
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                self.optimizer.step()
                eloss += [loss.cpu().detach().numpy()]

            if devpath:
                self.predict_all_txts_in_directory(devdocs, dev_tokens, output_dir='/wynton/protected/project/outcome_pred/n2c2-tr2-socdet/devpreds')
                predcorp = Corpus()
                predcorp.import_dir('/wynton/protected/project/outcome_pred/n2c2-tr2-socdet/devpreds')
                deveval = score_docs(devdocs, predcorp.docs(as_dict=True), \
                           labeled_args=SDOH_LABELED_ARGUMENTS,
                           score_trig=OVERLAP,
                           score_span=EXACT,
                           score_labeled=LABEL,
                           output_path=None,
                           include_detailed=False)
                print(deveval.loc[deveval['argument'].isin(['Trigger','OVERALL'])][['event','F1']].set_index('event'))

            epoch_time = time.time() - t0
            epoch_times.append(epoch_time)
            print('Epoch', epoch, '\tLoss:', numpy.mean(eloss), '\tTook:', round(epoch_time, 1), 's',
                  'Time left:', round(numpy.mean(epoch_times) * (max_num_epochs - epoch), 1), 's',
                  'ETA:', (datetime.datetime.now() + datetime.timedelta(
                    seconds=numpy.mean(epoch_times) * (max_num_epochs - epoch))).strftime('%H:%M on %d %b %Y '))

            epoch += 1

        self.eval()

    def forward(self, doc, bert_feats, gt_triggers=True, doc_id=None):
        # 1.1 encode text
        if not self.is_roberta:
            embedded_seq = self.word_embs(input_ids=bert_feats['input_ids'].to(self.device),
                                          attention_mask=bert_feats['attention_mask'].to(self.device),
                                          token_type_ids=bert_feats['token_type_ids'].to(self.device)
                                          )[0].squeeze() # TODO: Squeeze, unsqueeze may crash if batch size > 1?
        else:
            embedded_seq = self.word_embs(input_ids=bert_feats['input_ids'].to(self.device),
                                          attention_mask=bert_feats['attention_mask'].to(self.device),
                                          )[0].squeeze()  # TODO: Squeeze, unsqueeze may crash if batch size > 1?
        if self.emb_dropout:
            embedded_seq = self.emb_dropout_layer(embedded_seq)

        hidden_states, _ = self.rnn(embedded_seq.unsqueeze(dim=1))

        if self.use_attn:
            # Adding attention here before dropout
            attns = self.attn_layer(hidden_states)
            attns = torch.nn.functional.softmax(attns.squeeze(dim=2), dim=0)
            hidden_states = torch.tensordot(attns.unsqueeze(dim=1), hidden_states, dims=2).unsqueeze(dim=1)

        dropped_out_hidden_states = self.dropout_layer(hidden_states)

        # 2.1 get trigger scores
        if self.relus:
            raw_trigger_scores = torch.nn.functional.leaky_relu(self.trigger_layer(dropped_out_hidden_states))
        else:
            raw_trigger_scores = self.trigger_layer(dropped_out_hidden_states)

        trigger_scores = nn.functional.log_softmax(raw_trigger_scores, 2)  # output

        # 3.1 get labeled argument candidates
        if gt_triggers:
            trigger_tags, _ = self.get_gt_trigger_tags_from_doc(doc, bert_feats=bert_feats, doc_id=doc_id)

        else:
            trigger_tags = [self.bio_trigger_tags[i] for i in torch.argmax(trigger_scores, 2)]

        events = get_event_trigger_info_from_bio_tags(trigger_tags)

        # 3.2 predict labeled argument probs
        event_labels_and_arguments_scores = []
        for (etype, start, end) in events:
            # print(etype, start, end)
            # dist_vec = torch.tensor([max([start-i, 0, i-end]) for i in range(0,len(token_seq))]).view(len(token_seq),1,1)
            indicator_vec = torch.tensor([1 if i in range(start, end) else 0 for i in range(0,
                                                                                            bert_feats['input_ids'].size()[1])]
                                         ).view(bert_feats['input_ids'].size()[1], 1, 1).to(self.device)
            if self.divide_subtask_networks:
                hidden_states, _ = self.subtask_rnns[etype](embedded_seq.unsqueeze(dim=1))

                if self.use_attn:
                    # Adding attention here before dropout
                    attns = self.subtask_attn_layer(hidden_states)
                    attns = torch.nn.functional.softmax(attns, dim=0)
                    hidden_states = torch.tensordot(attns, hidden_states.squeeze(), dims=1)

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

    def loss(self, doc, bert_feats, trigger_scores, event_labels_and_arguments_scores, doc_id=None):
        losses = {}

        criterion = nn.NLLLoss()
        gt_trigger_tags, gt_subtask_tags = self.get_gt_trigger_tags_from_doc(doc, tag_indices=True,
                                                                             bert_mapping=True,
                                                                             bert_feats=bert_feats,
                                                                             doc_id=doc_id
                                                                             )

        losses['triggers'] = criterion(trigger_scores.view(len(gt_trigger_tags), len(self.bio_trigger_tags)).to(self.device),
                                       torch.tensor(gt_trigger_tags).to(self.device)).mean()

        losses['subtasks'] = 0
        for (espec, subtask_scores) in event_labels_and_arguments_scores:
            for subtask in subtask_scores:
                if espec in gt_subtask_tags:
                    subtask_in = subtask_scores[subtask].view(len(gt_subtask_tags[espec][subtask]),
                                                              len(self.bio_tags[espec[0]][subtask]))
                    losses['subtasks'] += criterion(subtask_in.to(self.device),
                                                    torch.tensor(gt_subtask_tags[espec][subtask]).to(self.device))
                    # print(espec, gt_subtask_tags[espec])
                # else:
                #     print('missing espec', espec)

        return losses['triggers'] + self.subtask_weight * losses['subtasks']

    def predict_all_txts_in_directory(self, testdocs, test_tokens, output_dir='predictions'):
        if output_dir:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)
            os.makedirs(output_dir)

        for doc_id, doc in testdocs.items():
            try:
                predicted_annotations = self.predict(doc, test_tokens[doc_id], doc_id=doc_id)
            except:
                print("Error in predictions on doc: ", doc_id)
                predicted_annotations = ''

            ann_out_path = output_dir + '/' + doc_id + '.ann'
            with open(ann_out_path, 'w') as f:
                f.write(predicted_annotations)
            # copy text
            txt_out_path = output_dir + '/' + doc_id + '.txt'
            with open(txt_out_path, 'w') as f:
                f.write(doc.text)

    def predict(self, doc, bert_feats, doc_id=None):
        #print('PRED', doc.id)
        bert_offsets = bert_feats.offset_mapping.squeeze()
        brat_offset = [tok_offset for sent in doc.token_offsets for tok_offset in sent]

        bert_offset_iter = iter(bert_offsets)
        brat_offset_iter = iter(brat_offset)

        offset_list = list()
        for bert_start, bert_end in bert_offset_iter:
            if bert_start == 0:
                offset_list.append(next(brat_offset_iter))
            else:
                offset_list.append(offset_list[-1])
        doctext = doc.text.replace('\n', ' ')

        # 1 forward on text docs
        trigger_scores, event_labels_and_arguments_scores = self.forward(doc,
                                                                         gt_triggers=False,
                                                                         bert_feats=bert_feats,
                                                                         doc_id=doc_id)

        trigger_tags = [self.bio_trigger_tags[i] for i in torch.argmax(trigger_scores, 2)]

        # trigger_tags = [('I'+trigger_tags[i-1][1:] if tag == 'SKIP' else tag) for i, tag in enumerate(trigger_tags)]

        # print("New trigger preds: ", trigger_tags)
        Ts, Es, As = [], [], []

        for ((etype, start, end), subtask_scores) in event_labels_and_arguments_scores:
            #print('>', etype, start, end)
            T_str = "T" + str(len(Ts)  + 1) + '\t' + etype + ' ' + str(offset_list[start][0]) + ' ' + str(
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
                    if a_type.split('-')[1] != 'NA':
                        #print('ARG', a_type.split('-')[1])
                        A_str = "A" + str(len(As) + 1) + '\t' + a_type.split('-')[0] + 'Val T' + str(len(Ts)) + ' ' + \
                                a_type.split('-')[1]
                        As.append(A_str)
            E_str = "E" + str(len(Es) + 1) + '\t' + ' '.join(arg_summaries)
            Es.append(E_str)

        ann = '\n'.join(Ts + Es + As)
        #print(ann)
        return ann


def load_model(model, model_loading_path):
    model.build()
    state_dict = torch.load(model_loading_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
    model.eval()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Script to train and test a SDOH model for the N2C2 Shared Task (2022).')
    parser.add_argument('-labeled_train_corpus', required=False, help='labeled_train_corpus_path', default=None)
    parser.add_argument('-second_labeled_train_corpus', required=False,
                        help='a second labeled_train_corpus_path (target)', default=None)
    parser.add_argument('-labeled_dev_corpus', required=False, help='labeled_dev_corpus_path', default=None)
    parser.add_argument('-selftrain_corpus', required=False, help='seltrain_corpus_path', default=None)
    parser.add_argument('-output_test_predictions', required=False, help='predictions_test_corpus_path', default="./predictions/")
    parser.add_argument('-unlabeled_test_corpus', required=False, help='unlabeled_test_corpus_path', default=None)
    parser.add_argument('-labeled_test_corpus', required=False, help='labeled_test_corpus_path', default=None)
    parser.add_argument('-model_loading_path', required=False, help='model_loading_path', default=None)
    parser.add_argument('-model_saving_path', required=False, help='model_saving_path', default="model.t")
    parser.add_argument('-max_num_epochs', required=False, help='max_num_epochs (default: 250)', type=int, default=250)
    parser.add_argument('-max_seq_len', required=False, help='max_num_epochs (default: 512)', type=int, default=512)
    parser.add_argument("--add_prefix_space", action="store_true", help="Add prefix space for roberta tokenizer")
    parser.add_argument('-freeze_emb', required=False, help='freeze_embeddings (default: True)', type=bool, default=False)
    parser.add_argument('-use_attn', required=False, help='use attention layer (default: False)', type=bool, default=False)
    parser.add_argument('-bert_model_name_or_path', required=False, help='bert model_name_or_path (default: clinicalbert)', type=str, default='emilyalsentzer/Bio_ClinicalBERT')
    parser.add_argument('-conflate_digits', required=False, help='conflate digits (flag: 1 or 0, default: 0)', type=int, default=0)
    parser.add_argument('-lowercase_tokens', required=False, help='lowercase tokens (flag: 1 or 0, default: 0)', type=int, default=0)
    parser.add_argument('-unk1_prob', required=False, help='replace 1 time occuring tokens sometimes with unk to train the unk embedding, default prob: 0.5', type=float, default=0.5)
    parser.add_argument('-wemb_size', required=False, help='word embedding size, default: 768', type=int, default=768)
    parser.add_argument('-rnn_dim', required=False, help='LSTM/GRU dimension, default: 100', type=int, default=100)
    parser.add_argument('-rnn_type', required=False, help='LSTM or GRU, default: LSTM', type=str, default='LSTM')
    parser.add_argument('-output_eval', required=False, help='Output evaluation file', type=str, default="evaluation.csv")
    parser.add_argument('-dropout', required=False, help='Dropout probability, default: 0.5', type=float, default=0.5)
    parser.add_argument('-lr', required=False, help='Learning rate, default: 0.001', type=float, default=0.001)
    parser.add_argument('-relus', required=False, help='Use Relu activations, default: 1', type=int, default=1)
    parser.add_argument('-emb_dropout', required=False, help='Use dropout on embedding layer, default: 0', type=float, default=0)
    parser.add_argument('-subtask_weight', required=False, help='Loss weight assigned to the subtask loss (0, inf), default: 1', type=float, default=1)
    parser.add_argument('-clip_gradient', required=False, help='Clip gradients during training, default: 5', type=float, default=5)
    parser.add_argument('-rnn_depth', required=False, help='Number of sequential RNN layers, default: 1', type=int, default=1)
    parser.add_argument('-divide_subtask_networks', required=False, help='Use separate RNNs for each subtask, default: 0', type=float, default=0)
    parser.add_argument('-loss_weight_train2', required=False,
                        help='Weight for the loss for documents in training set 2 (if provided), default: 1',
                        type=float, default=1)
    parser.add_argument('-calibrate', required=False, help='Calibrate the model on the target files, default: 0',
                        type=str, default=0)

    args = parser.parse_args()

    if args.add_prefix_space:
        is_roberta = True
    else:
        is_roberta = False

    # Define model
    model = sdoh_bert_model(bert_model_name_or_path=args.bert_model_name_or_path,
                            is_roberta=is_roberta,
                            max_seq_len=args.max_seq_len,
                            freeze_emb=args.freeze_emb,
                            use_attn=args.use_attn,
                            conflate_digits=args.conflate_digits,
                            lowercase=args.lowercase_tokens,
                            wemb_size=args.wemb_size,
                            rnn_dim=args.rnn_dim,
                            subtask_weight=args.subtask_weight,
                            dropout=args.dropout,
                            relus=args.relus,
                            emb_dropout=args.emb_dropout,
                            rnn_type=args.rnn_type,
                            divide_subtask_networks=args.divide_subtask_networks,
                            rnn_num_layers=args.rnn_depth)

    # Load a model
    if args.model_loading_path:
        model = load_model(model, args.model_loading_path)
    else:
        # Train and save the model
        if args.selftrain_corpus:
            model.selftrain_fit(selftrainpath=args.selftrain_corpus,
                                trainpath=args.labeled_train_corpus, trainpath2=args.second_labeled_train_corpus,
                                lossweight2=args.loss_weight_train2,
                                devpath=args.labeled_dev_corpus,
                                max_num_epochs=args.max_num_epochs, lr=args.lr, grad_clip=args.clip_gradient)
        else:
            model.fit(trainpath=args.labeled_train_corpus, trainpath2=args.second_labeled_train_corpus,
                      lossweight2=args.loss_weight_train2,
                      devpath=args.labeled_dev_corpus,
                      max_num_epochs=args.max_num_epochs, lr=args.lr, grad_clip=args.clip_gradient)

        if args.calibrate: # only refit bias terms in a given dataset
            model.calibrate(args.calibrate, max_num_epochs=args.max_num_epochs, lr=args.lr)

        torch.save(model.state_dict(), args.model_saving_path)

    # 2. Make predictions in the test set
    corpus = Corpus()
    corpus.import_dir(path=args.unlabeled_test_corpus)
    docs = corpus.docs(as_dict=True)
    doc_tokens = dict()
    for doc_id, doc in docs.items():
        tokens = [tok for sent in doc.tokens for tok in sent]
        doc_tokens[doc_id] = model.bert_tokenizer(tokens, is_split_into_words=True,
                                                 return_offsets_mapping=True,
                                                 padding=True,
                                                 truncation=True,
                                                 max_length=model.max_seq_len,
                                                 add_special_tokens=False,
                                                 return_tensors='pt')
    model.predict_all_txts_in_directory(docs, doc_tokens, args.output_test_predictions)

    # # 3. Evaluate the predictions
    if args.labeled_test_corpus:
        print('Evaluating in', args.labeled_test_corpus)
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

        micro_average_subtypes_csv(args.output_eval,
                                   '.'.join(args.output_eval.split('.')[:-1] + ['_microaveraged'] + ['.csv']))
        pandas.set_option("display.max_rows", 100, "display.max_columns", 100)
        pandas.options.display.width = 0
        print(evaluation)
