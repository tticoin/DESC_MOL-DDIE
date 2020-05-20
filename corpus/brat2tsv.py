import sys
import itertools
import glob

# Get instances from brat format files
# TSV format
# [sentences]\t[label]\t[drug1 name]\t[drug2 name]\n

if len(sys.argv) != 3:
    sys.stderr.write('Usage: python3 %s brat_dir base' % (sys.argv[0]))
    sys.exit(-1)

brat_dir = sys.argv[1]
tsv_out = open(sys.argv[2], 'w')

class Entity:
    def __init__(self, annline, mapline):
        tabs = annline.strip().split('\t')
        self.number = tabs[0]
        self.type = tabs[1].split()[0]
        self.offset_text = ' '.join(tabs[1].split()[1:]).split(';')
        self.offsets = [tuple([int(y) for y in x.split()]) for x in self.offset_text]
        self.surface = tabs[2]
        self.id = mapline.strip().split('\t')[1]

class Relation:
    def __init__(self, annline, mapline):
        tabs = annline.strip().split('\t')
        self.number = tabs[0]
        self.type = tabs[1].split()[0]
        self.pair = tuple([x.split(':')[1] for x in tabs[1].split()[1:]])
        self.id = mapline.strip().split('\t')[1]

class Span:
    def __init__(self, offset):
        self.l = offset[0]
        self.r = offset[1]
    def is_included(self, s):
        return s.l <= self.l and self.r <= s.r

# Choose one offset from separated offsets
def choose_offset(entities):
    for entity in entities:
        other_entities = [e for e in entities if e != entity]
        other_offsets = [e.offsets for e in other_entities]
        # Flatten list
        other_offsets = [item for sublist in other_offsets for item in sublist]

        valid_offsets = [o for o in entity.offsets]
        if len(entity.offsets) >= 2:
            for offset in entity.offsets:
                s = Span(offset)
                for other_offset in other_offsets:
                    other_s = Span(other_offset)
                    if s.is_included(other_s) or other_s.is_included(s):
                        valid_offsets.remove(offset)
                        break

        # Special case
        if len(valid_offsets) == 0:
            valid_offsets = [entity.offsets[1]]

        entity.offset = valid_offsets[0]

# Split entities into per sentence
def split_entities(entities, offsets):
    sent_entities_list = []
    prev_offset = 0
    for offset in offsets:
        sent_entities = [e for e in entities if prev_offset <= e.offset[1] <= offset]
        prev_offset = offset
        sent_entities_list.append(sent_entities)
    return sent_entities_list

# Replace entities into special tokens (DRUG1, DRUG2, DRUGOTHER)
def replace_entity(txt_list, replace_dict):
    offset = 0
    for k, v in sorted(replace_dict.items()):
        txt_list[k[0] + offset : k[1] + offset] = v
        offset += len(v) - (k[1] - k[0])

# Process one brat file
def main(base):
    with open(base + '.txt', 'r') as f:
        txt = f.read().strip()
    with open(base + '.ann', 'r') as f:
        annlines = f.readlines()
    with open(base + '.map', 'r') as f:
        maplines = f.readlines()

    entities = [Entity(x, y) for x, y in zip(annlines, maplines) if x[0] == 'T']
    relations = [Relation(x, y) for x, y in zip(annlines, maplines) if x[0] == 'R']

    choose_offset(entities)

    #foo = [e.offset for e in entities]
    for e in entities:
        other = [x for x in entities if x != e]
        s = Span(e.offset)
        for o in other:
            os = Span(o.offset)
            if s.is_included(os) or os.is_included(s):
                print('!')

    offset = 0
    offsets = []
    txtlines = txt.split('\n')
    for x in txtlines:
        offset += len(x) + 1
        offsets.append(offset)

    sent_entities = split_entities(entities, offsets)
    for i, s in enumerate(sent_entities):
        for e1, e2 in itertools.combinations(s, 2):
            other_entities = [x for x in s if x != e1 and x != e2]

            replace_dict = {}
            replace_dict[e1.offset] = ' DRUG1 '
            replace_dict[e2.offset] = ' DRUG2 '
            for other_e in other_entities:
                replace_dict[other_e.offset] = ' DRUGOTHER '

            txt_list = list(txt)
            replace_entity(txt_list, replace_dict)
            sent = ''.join(txt_list).split('\n')[i]

            label = 'negative'
            for r in relations:
                if r.pair[0] == e1.number and r.pair[1] == e2.number:
                    label = r.type 
                    break
            
            tsv_out.write('{}\t{}\t{}\t{}\n'.format(sent, label, e1.surface, e2.surface))
            #tsv_out.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(sent, label, e1.surface, e2.surface, e1.id, e2.id))

for path in glob.glob(sys.argv[1] + '/*.txt'):
    main(path.replace('.txt', ''))

tsv_out.close()
