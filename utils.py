
import os
import json
import tarfile
import codecs
import sys
import shutil
import tempfile
import gc
import pickle


def dir_tree_run(action, base_dir):
    """
    Apply funcition "action" to the individual files from tree directory
    """
    _temp_f_name = ""
    for f_name in os.listdir(base_dir):
        _temp_f_name = os.path.join(base_dir,f_name)
        if os.path.isdir(_temp_f_name):
            dir_tree_run(action,_temp_f_name)
        else:
            action(_temp_f_name)

def process_open_xml(proc_id, xml_files, output_dir, filter_f = None):
    import pubmed_parser as pp
    
    def filter_mesh(string):
        return " ".join(map(lambda y:y[0], map(lambda x: x.split(";"), string.split(":")[1:])))
    
    print("[Process-{}] Started".format(proc_id))
    articles = []
    for file_name in xml_files:
        print(proc_id, file_name)
        try:
            articles.extend(pp.parse_medline_xml(file_name, year_info_only=False, nlm_category=False))
        except etree.XMLSyntaxError:
            print("Error on File " + file_name)
        
        gc.collect()
    
    if filter_f is not None:
        articles = filter(filter_f, articles)

    articles_mapped = list(map(lambda x:{"id":x["pmid"],
                                         "title":x["title"],
                                         "abstract":x["abstract"],
                                         "keywords":x["keywords"],
                                         "pubdate":x["pubdate"],
                                         "mesh_terms":filter_mesh(x["mesh_terms"]),
                                         "author":x["author"],
                                         "affiliation":x["affiliation"],}
                               ,articles))

    file_name = output_dir+"/TREC-PM-Baseline-{0:03}.p".format(proc_id)
    print("[Process-{}]: Store {}".format(proc_id, file_name))

    with open(file_name, "wb") as f:
        pickle.dump(articles_mapped, f)

    del articles
    print("[Process-{}] Ended".format(proc_id))

def multiprocess_xml_to_json(xml_files, n_process, max_store_size=int(6e6), store_path="/backup/TREC-PM/Corpus/collection-json.tar.gz"):
    from multiprocessing import Process
    
    total_files = len(xml_files)
    itter = total_files//n_process
    
    tmp_path = tempfile.mkdtemp()
    process = []
    
    try:
        
        for _i,i in enumerate(range(0, total_files, itter)):
            process.append(Process(target=process_open_xml, args=(_i, xml_files[i:i+itter], tmp_path)))

        print("[MULTIPROCESS LOOP] Starting", n_process, "process")
        for p in process:
            p.start()

        print("[MULTIPROCESS LOOP] Wait", n_process, "process")
        for p in process:
            p.join()

        del process
        gc.collect()
           
        ## merge 
        resulting_files = sorted(os.listdir(tmp_path))
        articles = []
        for file in resulting_files:
            with open(os.path.join(tmp_path, file), "rb") as f:
                articles.extend(pickle.load(f))

        # batch save
        
        size = len(articles)
        print(size)
        itter = max_store_size
        
        f_names = []
        
        for i in range(0, size, itter):
            file_name = tmp_path+"/TREC-PM-baseline-{0:08}-to-{1:08}".format(i, min(size, i+itter))
            print("Save file",file_name,":",end="")
            json.dump(articles[i:i+itter], open(file_name,"w"))
            f_names.append(file_name)
            print("Done")

        print("Start the compression")
        # build tar file
        with tarfile.open(store_path, "w:gz") as tar:
            for name in f_names:
                tar.add(name)

    except Exception as e:
        raise e

    finally:
        shutil.rmtree(tmp_path)
        
        
def collection_iterator(file_name, f_map=None):
    return collection_iterator_fn(file_name=file_name, f_map=f_map)()

def collection_iterator_fn(file_name, f_map=None):
    
    reader = codecs.getreader("ascii")
    tar = tarfile.open(file_name)

    print("[CORPORA] Openning tar file", file_name)

    members = tar.getmembers()
    
    def generator():
        for m in members:
            print("[CORPORA] Openning tar file {}".format(m.name))
            f = tar.extractfile(m)
            articles = json.load(reader(f))
            if f_map is not None:
                articles = list(map(f_map, articles))
            yield articles
            f.close()
            del f
            gc.collect()
    return generator