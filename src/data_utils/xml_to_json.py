import xml.etree.ElementTree as ET
from tqdm import tqdm
import libarchive.public
import json


def default_archive_to_json(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = input_dir
    for table, no_write in [
        ('Users', ['AboutMe']),
        ('Posts', ['Body']),
        ('PostHistory', ['Text']),
        ('Votes', []),
    ]:
        archive_to_json(
            archive = f'{input_dir}/stackoverflow.com-{table}.7z',
            json_file = f'{output_dir}/{table}.json',
            no_write = no_write,
        )


def archive_to_json(archive, json_file, no_write):
    with open(json_file, 'w') as f:
        for d in tqdm(load_archive(archive), desc=archive):
            for c in no_write:
                if c in d:
                    del d[c]
            f.write(json.dumps(d) + '\n')


def load_archive(archive):
    with libarchive.public.file_reader(archive) as e:
        entry = next(e)
        buf = b''
        for block in entry.get_blocks():
            buf += block
            try:
                ss = buf.decode('utf8').split('\n')
            except UnicodeDecodeError:
                continue
            for s in ss[:-1]:
                d = parse(s)
                if d is not None:
                    yield d
            buf = ss[-1].encode('utf8')


int_lim = 1<<63
def try_int(dic):
    for k in dic:
        try:
            dic[k] = int(dic[k]) if -int_lim < int(dic[k]) < int_lim else dic[k]
        except Exception:
            continue
    return dic


def parse(line):
    try:
        return try_int(ET.fromstring(line).attrib)
    except ET.ParseError:
        return None


if __name__ == "__main__":
    in_folder = "/Volumes/Seagate Backup Plus Drive/"
    default_archive_to_json("")