import uproot

file_path = "./conex_eposlhc_000000001_100.root"
ntuple = uproot.open(file_path)
tree = ntuple["Shower"]
