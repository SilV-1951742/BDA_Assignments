from functools import partial
import re
import xml.etree.ElementTree as ET

entry_regex = re.compile(r"<(article|book|phdthesis|www|incollection)[\s\S]*?<(\/article|\book|phdthesis|www|incollection)>")

def entry_string(filename, chunk_size):
    with open(filename, 'r') as f:
        previous = ""
        start_prev = 0
        file_r = partial(f.read, chunk_size)
        
        for data in iter(file_r, ""):
            data = previous + data
            for result in re.finditer(entry_regex, data):
                start_prev = result.end()
                print(result.group(0))
                yield result.group(0)
            previous = data[start_prev:]

def entry_tree(xml_string):
    return ET.fromstring(xml_string)

        
            
def main():
    gen_entry_string =  entry_string("dblp.xml", 1*1024*1024)

    for i in range(20):
        print(f"Entry {i}: \r\n {next(gen_entry_string)}\r\n\r\n\r\n")
        root = entry_tree(next(gen_entry_string))
        print(f"Root tag {root.tag}, attrib {root.attrib}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()

