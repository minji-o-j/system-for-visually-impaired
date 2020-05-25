import os
import xml.etree.ElementTree as ET
path="./"
file_list=os.listdir(path)
file_list_xml=[file for file in file_list if file.endswith(".xml")]
i=0

for file in file_list_xml:
  tree=ET.parse(file)
  root=tree.getroot()
  path_tag=root.find("path")
  path_tag.text=""
  tree.write(file+"out.xml",encoding="utf-8",xml_declaration=True)
 
 