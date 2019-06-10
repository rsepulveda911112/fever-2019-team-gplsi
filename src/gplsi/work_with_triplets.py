# coding: utf8

"""
Utilizar directamente los dos ultimos metodos
------------------
"""
import nltk.data
import spacy
import neuralcoref
from statistics import mean
import operator
#from spacy.vocab import Vocab

SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
OBJECTS = ["dobj", "dative", "attr", "oprd"]

"""
NLP, Concepts
"""
nlp = spacy.load('en_core_web_lg')
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
coref = neuralcoref.NeuralCoref(nlp.vocab)
nlp.add_pipe(coref, name='neuralcoref')

def split_sentences(text):
    """
    Recibe un texto y devuelve un listado de oraciones.
    """

    result_sents = []

    text_new = " ".join(text.split())
    sentences = sent_detector.sentences_from_text(text_new)
    for s in sentences:
        s = s.strip()
        if s != "":
            result_sents.append(s)

    return result_sents


class Concept:
    def __init__(self, text, label, pos_init, normalized=None):
        self.text = text
        self.label = label
        self.pos_init = pos_init
        self.pos_end = pos_init + len(text)
        self.normalized = normalized or text

    def __and__(self, other):
        return max(0, min(self.pos_end, other.pos_end) - max(self.pos_init, other.pos_init)) > 0

    def __repr__(self):
        return self.text

    def __json__(self):
        return self.__dict__


"""
Entity Sensor
"""

class EntitySensor:
    """
    Este sensor recibe un texto de entrada,
    y devuelve una lista de entidades extraÃ­das del mismo.

    Las entidades son tuplas de la forma
    `(texto, tipo)`.
    """
    def run(self, text):

        # Obtener todos los Ã¡rboles y acciones
        entities = []

        for e in nlp(text).ents:
            entities.append(Concept(e.text, e.label_, e.start_char))

        return entities

"""
EntitySensorCarmen
"""
class EntitySensorCarmen:
    """
    Este sensor recibe un texto de entrada,
    y devuelve una lista de entidades extraÃ­das del mismo.

    Las entidades son tuplas de la forma
    `(texto, tipo)`.
    """
    def run(self, text):

        # Obtener todos los Ã¡rboles y acciones
        entities = []

        for e in nlp(text).ents:
            entities.append(Concept(e.text, e.label_, e.start_char))

        return entities

"""
CoreferenceSensor
"""

def resolve_coref(text):
    doc = nlp(u'{0}'.format(text))
    return doc._.coref_resolved

class CoreferenceSensor:
    """
    Devuelve una lista de coreferencias, en forma
    de lista de pares [ (Concept, Concept) ] donde
    el primero es una referencia al segundo.
    """
    def __init__(self):
        pass

    def run(self, text):
        for k,v in coref(text).items():
            yield (
                Concept(k.text, None, k.start_char),
                Concept(v.text, None, v.start_char),
            )


class ActionSensor:
    """
   This sensor receives an input text,
    and returns a list of extracted actions
    of the same.

      Actions are tuples of the form
    (action, subject, object).
    """
    def run(self, text):
        for sentence in nlp(text).sents:
            for action in self.extract_actions(sentence):
                yield action

    def get_subs_from_conjunctions(self, subs):
        more_subs = []
        for sub in subs:
            # rights is a generator
            rights = list(sub.rights)
            rightDeps = {tok.lower_ for tok in rights}
            if "and" in rightDeps:
                more_subs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == "NOUN"])
                if len(more_subs) > 0:
                    more_subs.extend(self.get_subs_from_conjunctions(more_subs))
        return more_subs

    def get_objs_from_conjunctions(self, objs):
        moreObjs = []
        for obj in objs:
            # rights is a generator
            rights = list(obj.rights)
            rightDeps = {tok.lower_ for tok in rights}
            if "and" in rightDeps:
                moreObjs.extend([tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ == "NOUN"])
                if len(moreObjs) > 0:
                    moreObjs.extend(self.get_objs_from_conjunctions(moreObjs))
        return moreObjs

    def get_verbs_from_conjunctions(self, verbs):
        more_verbs = []
        for verb in verbs:
            rightDeps = {tok.lower_ for tok in verb.rights}
            if "and" in rightDeps:
                more_verbs.extend([tok for tok in verb.rights if tok.pos_ == "VERB"])
                if len(more_verbs) > 0:
                    more_verbs.extend(self.get_verbs_from_conjunctions(more_verbs))
        return more_verbs

    def find_subs(self, tok):
        head = tok.head
        while head.pos_ != "VERB" and head.pos_ != "NOUN" and head.head != head:
            head = head.head
        if head.pos_ == "VERB":
            subs = [tok for tok in head.lefts if tok.dep_ == "SUB"]
            if len(subs) > 0:
                verb_negated = self.is_negated(head)
                subs.extend(self.get_subs_from_conjunctions(subs))
                return subs, verb_negated
            elif head.head != head:
                return self.find_subs(head)
        elif head.pos_ == "NOUN":
            return [head], self.is_negated(tok)
        return [], False

    def is_negated(self, tok):
        negations = {"no", "not", "n't", "never", "none"}
        for dep in list(tok.lefts) + list(tok.rights):
            if dep.lower_ in negations:
                return True
        return False

    def get_objs_from_prepositions(self, deps):
        objs = []
        for dep in deps:
            if dep.pos_ == "ADP" and dep.dep_ == "prep":
                objs.extend([tok for tok in dep.rights if tok.dep_  in OBJECTS or (tok.pos_ == "PRON" and tok.lower_ == "me")])
        return objs

    def get_objs_from_attrs(self, deps):
        for dep in deps:
            if dep.pos_ == "NOUN" and dep.dep_ == "attr":
                verbs = [tok for tok in dep.rights if tok.pos_ == "VERB"]
                if len(verbs) > 0:
                    for v in verbs:
                        rights = list(v.rights)
                        objs = [tok for tok in rights if tok.dep_ in OBJECTS]
                        objs.extend(self.get_objs_from_prepositions(rights))
                        if len(objs) > 0:
                            return v, objs
        return None, None

    def get_obj_from_xcomp(self, deps):
        for dep in deps:
            if dep.pos_ == "VERB" and dep.dep_ == "xcomp":
                v = dep
                rights = list(v.rights)
                objs = [tok for tok in rights if tok.dep_ in OBJECTS]
                objs.extend(self.get_objs_from_prepositions(rights))
                if len(objs) > 0:
                    return v, objs
        return None, None

    def get_all_subs(self, v):
        verbNegated = self.is_negated(v)
        subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
        if len(subs) > 0:
            subs.extend(self.get_subs_from_conjunctions(subs))
        else:
            foundSubs, verbNegated = self.find_subs(v)
            subs.extend(foundSubs)
        return subs, verbNegated

    def get_all_objs(self, v):
        # rights is a generator
        rights = list(v.rights)
        objs = [tok for tok in rights if tok.dep_ in OBJECTS]
        objs.extend(self.get_objs_from_prepositions(rights))

        potential_new_verb, potential_new_objs = self.get_obj_from_xcomp(rights)
        if potential_new_verb is not None and potential_new_objs is not None and len(potential_new_objs) > 0:
            objs.extend(potential_new_objs)
            v = potential_new_verb
        if len(objs) > 0:
            objs.extend(self.get_objs_from_conjunctions(objs))
        return v, objs

    def extract_actions(self, tokens):
        verbs = [tok for tok in tokens if tok.pos_ == "VERB" and tok.dep_ != "aux"]
        for v in verbs:
            subs, verb_negated = self.get_all_subs(v)

            #if verb_negated:
             #   continue

            if len(subs) > 0:
                v, objs = self.get_all_objs(v)
                for sub in subs:
                    subNegated = self.is_negated(sub)
                    action = Concept(text=v.text, label="action", pos_init=v.idx, normalized=v.lemma_)
                    subj = Concept(text=sub.text, label="subject", pos_init=sub.idx)
                    if len(objs)>0:
                        for obj in objs:
                            objNegated = self.is_negated(obj)
                            obj = Concept(text = obj.text, label = "object", pos_init = obj.idx)
                    else:
                        objNegated = False
                        obj=Concept(text='',label="object",pos_init=0)
                    yield (action, subj, obj,verb_negated,objNegated,subNegated)

    def findSVOs(self,tokens):
        svos = []
        verbs = [tok for tok in tokens if tok.pos_ == "VERB" and tok.dep_ != "aux"]
        for v in verbs:
            subs, verbNegated = self.get_all_subs(v)
            # hopefully there are subs, if not, don't examine this verb any longer
            if len(subs) > 0:
                v, objs = self.get_all_objs(v)
                for sub in subs:
                    for obj in objs:
                        objNegated = self.is_negated(obj)
                        svos.append((sub.lower_, "!" + v.lower_ if verbNegated or objNegated else v.lower_, obj.lower_))

        return svos


class My_Sensor:
    def __init__(self):
        self.actions = ActionSensor()
        self.entities = EntitySensor()
        self.coreferences = CoreferenceSensor()


    def run_file(self, filename):
        with open(filename) as fp:
            return list(self.run(fp.read()))

    '''
    the Extract_triplets method returns an ienumerable of tuples with the following parameters:
     action
     subject
     object
     negation in the verb (yes/no)
     Subject Negation (yes/no)
     negation in the object (yes/no)

     '''

    def extract_triplets(self, text):


        #corefs = list(self.coreferences.run(text))
        coref_resolved_text=resolve_coref(text)
        entities = list(self.entities.run(text))

        for action, subj, obj,verb_negated,objNegated,subNegated in self.actions.run(coref_resolved_text):
            subj.label = dict(role="subject", entity="NONE")
            obj.label = dict(role="object", entity="NONE")

            for e in entities:
                if not e.label:
                    continue

                if e & subj:
                    subj.label['entity'] = e.label
                    e.label = subj.label
                    subj = e
                if e & obj:
                    obj.label['entity'] = e.label
                    e.label = obj.label
                    obj = e

            yield (action, subj, obj,verb_negated,subNegated,objNegated)


    '''
     the Extract_triplets method returns an ienumerable of tuples with the following parameters:
     action
     subject
     object
     negation in the verb (yes/no)
     Subject Negation (yes/no)
     negation in the object (yes/no)

     '''
def extract_triplets(text):
    my_sensor=My_Sensor()
    return my_sensor.extract_triplets(text)


    '''
    the Compare_triplets method returns:
    similarity between actions
    similarity between subjects
    similarity of objects
    phrase similarity
    negation in the verb of the first triplet (yes/no)
    negation in the subject of the first triplet (yes/no)
    negation in the object of the first triplet (yes/no)
    negation in the verb of the second triplet (yes/no)
    negation in the subject of the second triplet (yes/no)
    negation in the object of the second triplet (yes/no)

    '''

def compare_triplets(s1,s2):
    token_subject1 = nlp(u'{0}'.format(str.lower(s1[1].normalized)))
    token_subject2 = nlp(u'{0}'.format(str.lower(s2[1].normalized)))
    token_object1 = nlp(u'{0}'.format(str.lower(s1[2].normalized)))
    token_object2 = nlp(u'{0}'.format(str.lower(s2[2].normalized)))
    token_action1 = nlp(u'{0}'.format(str.lower(s1[0].normalized)))
    token_action2 = nlp(u'{0}'.format(str.lower(s2[0].normalized)))
    token_triplet1 = nlp(u'{0}'.format(str.lower(s1[0].normalized+' '+s1[1].normalized+' '+s1[2].normalized)))
    token_triplet2 = nlp(u'{0}'.format(str.lower(s2[0].normalized+' '+s2[1].normalized+' '+s2[2].normalized)))

    subjects_similarity1 = token_subject1.similarity(token_subject2)
    objects_similarity1 = token_object1.similarity(token_object2)
    actions_similarity1 = token_action1.similarity(token_action2)
    triplets_similarity1 = token_triplet1.similarity(token_triplet2)

    subjects_similarity2 = token_subject2.similarity(token_subject1)
    objects_similarity2 = token_object2.similarity(token_object1)
    actions_similarity2 = token_action2.similarity(token_action1)
    triplets_similarity2 = token_triplet2.similarity(token_triplet1)

    subjects_similarity = max(subjects_similarity1, subjects_similarity2)
    objects_similarity = max(objects_similarity1, objects_similarity2)
    actions_similarity = max(actions_similarity1, actions_similarity2)
    triplets_similarity = max(triplets_similarity1, triplets_similarity2)

    return (
    actions_similarity, subjects_similarity, objects_similarity, triplets_similarity, s1[3], s1[4], s1[5], s2[3], s2[4],
    s2[5])




def get_scores_based_in_triplets(claim,sentences):
    claim_triplets= list(extract_triplets(claim))
    result=sentences
    mylist=[]
    count=0
    for s in sentences:
        triplets=list(extract_triplets(s))
        max_avg=0
        max_min_partial_similarity=0
        contradictions=0
        for t1 in claim_triplets:
            for t2 in triplets:
                similarities=compare_triplets(t1,t2)
                avg= mean([similarities[0],similarities[1],similarities[2]])
                if similarities[4]!=similarities[7]:
                    contradictions=-1
                tempmin=min(similarities[0],similarities[1],similarities[2])
                if avg>max_avg and tempmin>=max_min_partial_similarity  :
                    max_avg=avg
                    max_min_partial_similarity= tempmin
        mylist.append((round(max_avg,2),round(max_min_partial_similarity,2),contradictions,count))
        count=count+1
    mylist.sort(key = operator.itemgetter(1, 0),reverse=True)
    length=len(mylist)
    for x in range (0,length):
        result[mylist[x][3]]=length-x
    return result



def get_final_score(athene_predictions,triplets_score):
    athene_score=[]
    athene_final_score = []
    result=[]
    for p in range(0,len(athene_predictions)):
        athene_score.append((athene_predictions[p], len(athene_predictions)-p))

    for i in range(0,len(triplets_score)):
        athene_final_score.append((athene_score[i][0], athene_score[i][1]+triplets_score[i]))

    athene_final_score.sort(key = operator.itemgetter(1),reverse=True)
    for p in athene_final_score:
        result.append(p[0])
    return result