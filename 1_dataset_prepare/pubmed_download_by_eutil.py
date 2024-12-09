import subprocess
import argparse
import tempfile
import lxml.etree as ET
import pandas as pd
from tqdm import tqdm


def get_cmd(query, path_tsv):
    '''Generate the query command
    '''
    cmd = f"""
    esearch -db pubmed -query "{query}" | \
    efetch -format xml | \
    xtract -pattern PubmedArticle \
        -element MedlineCitation/PMID,ArticleTitle,Journal/Title,PubDate/Year \
        -block Abstract -sep " " -element AbstractText \
        -block MeshHeadingList -sep ";" -element DescriptorName \
        -block MeshHeadingList -sep ";" -element QualifierName \
    | (echo "pmid\ttitle\tjournal\tyear\tabstract\tmesh_terms\tmesh_term_major_topics"; cat) \
    > {path_tsv}
    """
    return cmd


def download_xml_by_query(query, path_xml):
    '''Download the raw XML file from PubMed by a given query
    '''
    cmd = f"""esearch -db pubmed -query "{query}" | \
    efetch -format xml > {path_xml}
    """
    print("* downloading XML")
    print(cmd)
    result = subprocess.run(cmd, shell=True)
    return result
    

def xtract_title(path_xml, path_out):
    '''Extract title from a given XML
    '''
    cmd = f"""cat {path_xml} | \
    xtract -pattern PubmedArticle \
        -element MedlineCitation/PMID,ArticleTitle | \
    (echo "pmid\ttitle"; cat) > {path_out}
    """
    print("* xtracting title")
    print(cmd)
    result = subprocess.run(cmd, shell=True)
    return result
    

def xtract_journal(path_xml, path_out):
    '''Extract journal from a given XML
    '''
    cmd = f"""cat {path_xml} | \
    xtract -pattern PubmedArticle \
        -element MedlineCitation/PMID,Journal/Title | \
    (echo "pmid\tjournal"; cat) > {path_out}
    """
    print("* xtracting journal")
    print(cmd)
    result = subprocess.run(cmd, shell=True)
    return result
    

def xtract_year(path_xml, path_out):
    '''Extract year from a given XML
    '''
    cmd = f"""cat {path_xml} | \
    xtract -pattern PubmedArticle \
        -element MedlineCitation/PMID,PubDate/Year | \
    (echo "pmid\tyear"; cat) > {path_out}
    """
    print("* xtracting year")
    print(cmd)
    result = subprocess.run(cmd, shell=True)
    return result
    

def xtract_abstract(path_xml, path_out):
    '''Extract abstract from a given XML
    '''
    cmd = f"""cat {path_xml} | \
    xtract -pattern PubmedArticle \
        -element MedlineCitation/PMID \
        -block Abstract -sep " " -element AbstractText | \
    (echo "pmid\tabstract"; cat) > {path_out}
    """
    print("* xtracting abstract")
    print(cmd)
    result = subprocess.run(cmd, shell=True)
    return result
    

def xtract_conclusions(path_xml, path_out):
    '''Extract conclusions from a given XML
    '''
    print("* xtracting conclusions by ET")
    parser = ET.XMLParser(recover=True)
    tree = ET.parse(path_xml, parser=parser)

    # with open(path_xml, 'r') as f:
    #     tree = ET.parse(f)
    root = tree.getroot()

    pubmed_data = []
    pubmed_articles = root.findall(".//PubmedArticle")
    print('* found %s pubmed_articles' % (len(pubmed_articles)))

    for pubmed_article in tqdm(pubmed_articles):
        pmid = pubmed_article.findtext(".//PMID")
        pubmed_entry = {
            'pmid': pmid,
            'conclusions': ''
        }

        abst_elements = pubmed_article.findall(".//AbstractText")
        conclusions = []
        for abst in abst_elements:
            if abst.get('NlmCategory') == 'CONCLUSIONS':
                conclusions.append(abst.text)

        pubmed_entry["conclusions"] = ' '.join(conclusions)

        # ok, we have got the information for this paper
        pubmed_data.append(pubmed_entry)

    # convert to df
    df = pd.DataFrame(pubmed_data)
    df.to_csv(
        path_out,
        sep='\t',
        index=False
    )
    print('* extracted conclusions into %s' % path_out)
    
    return df
    

def xtract_mesh_terms(path_xml, path_out):
    '''Extract mesh terms from a given XML
    '''
    cmd = f"""cat {path_xml} | \
    xtract -pattern PubmedArticle \
        -element MedlineCitation/PMID \
        -block MeshHeadingList -sep ";" -element DescriptorName | \
    (echo "pmid\tmesh_terms"; cat) > {path_out}
    """
    print("* xtracting mesh_terms")
    print(cmd)
    result = subprocess.run(cmd, shell=True)
    return result
    

def xtract_mesh_topics(path_xml, path_out):
    '''Extract mesh_topics from a given XML
    '''
    cmd = f"""cat {path_xml} | \
    xtract -pattern PubmedArticle \
        -element MedlineCitation/PMID \
        -block MeshHeadingList -sep ";" -element QualifierName | \
    (echo "pmid\tmesh_topics"; cat) > {path_out}
    """
    print("* xtracting mesh_topics")
    print(cmd)
    result = subprocess.run(cmd, shell=True)
    return result


def parse_xml(path_xml, path_tsv, flag_debug=False):
    '''Parse a XML file of pubmed data
    '''
    all_paths = []

    # second, get the title
    with tempfile.NamedTemporaryFile(delete=False) as f: path_title = f.name
    if flag_debug: path_title = 'raw_title.tsv'
    all_paths.append(path_title)
    xtract_title(path_xml, path_title)

    # third, get the journal
    with tempfile.NamedTemporaryFile(delete=False) as f: path_journal = f.name
    if flag_debug: path_journal = 'raw_journal.tsv'
    all_paths.append(path_journal)
    xtract_journal(path_xml, path_journal)

    # then, get year
    with tempfile.NamedTemporaryFile(delete=False) as f: path_year = f.name
    if flag_debug: path_year = 'raw_year.tsv'
    all_paths.append(path_year)
    xtract_year(path_xml, path_year)

    # then, get abstract
    with tempfile.NamedTemporaryFile(delete=False) as f: path_abstract = f.name
    if flag_debug: path_abstract = 'raw_abstract.tsv'
    all_paths.append(path_abstract)
    xtract_abstract(path_xml, path_abstract)

    # then, get mesh terms
    with tempfile.NamedTemporaryFile(delete=False) as f: path_mesh_terms = f.name
    if flag_debug: path_mesh_terms = 'raw_mesh_terms.tsv'
    all_paths.append(path_mesh_terms)
    xtract_mesh_terms(path_xml, path_mesh_terms)

    # then, get mesh topics
    with tempfile.NamedTemporaryFile(delete=False) as f: path_mesh_topics = f.name
    if flag_debug: path_mesh_topics = 'raw_mesh_topics.tsv'
    xtract_mesh_topics(path_xml, path_mesh_topics)
    all_paths.append(path_mesh_topics)
    xtract_mesh_terms(path_xml, path_mesh_terms)

    # then, get conclusions
    with tempfile.NamedTemporaryFile(delete=False) as f: path_conclusions = f.name
    if flag_debug: path_conclusions = 'raw_conclusions.tsv'
    all_paths.append(path_conclusions)
    xtract_conclusions(path_xml, path_conclusions)

    df = pd.read_csv(
        all_paths[0],
        sep='\t'
    )
    for path in all_paths[1:]:
        df_tmp = pd.read_csv(
            path,
            sep='\t'
        )
        df = pd.merge(df, df_tmp, on="pmid")
        print('* merged the %s' % path)

    # save the final df to given path
    df.to_csv(
        path_tsv,
        sep='\t',
        index=False
    )
    print('* saved the final df to %s' % path_tsv)

    return path_tsv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download PubMed data by query")
    parser.add_argument("-act", default="download", help="What to do? download/parse")
    parser.add_argument("-query", default="", help="PubMed format query")
    parser.add_argument("-path_xml", help="Path to the output xml file")
    parser.add_argument("-path_tsv", default="", help="Path to the output tsv file")
    parser.add_argument("--debug", default=False, type=bool, help="True or False")

    args = parser.parse_args()

    if args.act == 'download':
        download_xml_by_query(
            args.query, 
            args.path_xml
        )

    if args.act == 'parse':
        parse_xml(
            args.path_xml,
            args.path_tsv,
            args.debug
        )

    print('* done!')
