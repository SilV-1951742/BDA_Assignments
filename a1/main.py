from functools import partial
import re
import xml.etree.ElementTree as ET
from html import unescape


entry_regex = re.compile(
    r"<(article|book|phdthesis|www|incollection|proceedings|inproceedings)[\s\S]*?<(\/article|\/book|\/phdthesis|\/www|\/incollection|\/proceedings|\/inproceedings)>"
)


def entry_string(filename, chunk_size):
    with open(filename, "r") as f:
        previous = ""
        start_prev = 0
        file_r = partial(f.read, chunk_size)

        for data in iter(file_r, ""):
            data = previous + data
            for result in re.finditer(entry_regex, data):
                start_prev = result.end()
                yield result.group(0)
            previous = data[start_prev:]


def entry_tree(xml_string):
    return ET.fromstring(xml_string.replace("&", ""))


def main():
    gen_entry_string = entry_string("dblp.xml", 1 * 1024 * 1024)

    for entry in gen_entry_string:
        tmp_entry = unescape(entry)
        print(f"{tmp_entry}\r\n\r\n\r\n")
        root = entry_tree(tmp_entry)
        print(f"Root tag {root.tag}, attrib {root.attrib}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
