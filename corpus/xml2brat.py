import sys
import xml.etree.ElementTree as et

# Convert xml files to brat format (txt and ann files)

if len(sys.argv) != 3:
    sys.stderr.write('Usage: python3 %s xml base' % (sys.argv[0]))
    sys.exit(-1)

txt_out = open(sys.argv[2]+'.txt', 'w')
ann_out = open(sys.argv[2]+'.ann', 'w')
map_out = open(sys.argv[2]+'.map', 'w')

t_count = 0
r_count = 0
global_offset = 0
entity_dict = {} # key: number of entity, value: id of entity

def get_offset(s):
    # Convert xml format offset to brat format offset
    
    if len(s.split(';')) > 1: # separated entities
        return ';'.join([get_offset(x) for x in s.split(';')])
    else:
        sp = s.split('-')
        sp[1] = str(int(sp[1])+1) # adjust end offset for brat format
        return ' '.join([str(int(x)+global_offset) for x in sp])

tree = et.parse(sys.argv[1])
root = tree.getroot()

for sentence in root:
    text = sentence.attrib['text'].strip()
    txt_out.write('{}\n'.format(text))

    entities = sentence.findall('entity')
    pairs = sentence.findall('pair')

    for entity in entities:
        t_count += 1
        offset = get_offset(entity.attrib['charOffset'])
        ann_out.write('T{}\t{} {}\t{}\n'.format(t_count, entity.attrib['type'], offset, entity.attrib['text']))
        map_out.write('T{}\t{}\n'.format(t_count, entity.attrib['id']))
        entity_dict[entity.attrib['id']] = 'T{}'.format(t_count)
    global_offset += len(text) + 1 # add offset of linefeed

    for pair in pairs:
        if pair.attrib['ddi'] == 'true':
            r_count += 1
            try:
                ann_out.write('R{}\t{} Arg1:{} Arg2:{}\n'.format(r_count, pair.attrib['type'], entity_dict[pair.attrib['e1']], entity_dict[pair.attrib['e2']]))
            except:
                print(sys.argv[2])
            map_out.write('R{}\t{}\n'.format(r_count, pair.attrib['id']))

txt_out.close()
ann_out.close()
map_out.close()
