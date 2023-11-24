import numpy as np
from tqdm import tqdm
def main():
    #store the files into db
    # Post-hoc analysis: get topic terms, their relns, args, and document id in which they occur.
    df = pd.read_csv('parsedDoh.csv')
    topic_ids = df_topics.topic_id.unique().tolist()
    df_topics = pd.read_csv('topicsResults_sample_15.csv')
    topic_ids = df_topics.topic_id.unique().tolist()
    topic_doc_df = pd.read_csv('topicsDocProbabilities.csv')
    ls=[]
    
    for tid in tqdm(topic_ids):
        print('*** topic id: {0} ***'.format(tid))
        print('* top terms {0}'.format(' ,'.join((df_topics[df_topics.topic_id==tid].word).tolist())))
        words = df_topics.loc[df_topics.topic_id==tid].word.tolist()
        for w in words:
            #get reln and args
            relns = df_topics.loc[(df_topics.topic_id==tid) & (df_topics.word==w)].relns.str.split(',').tolist()[0]
            arg2 = df_topics.loc[(df_topics.topic_id==tid) & (df_topics.word==w)].arg2.str.split(',').tolist()[0]
            print('* w : ', w)
            print(' *** Realns {1}\n'.format(w, relns))
            print(' *** Args {1}\n'.format(w, arg2))
            #get document ids for these words' in reln and with arg2....
            ls = []
            for reln in relns:
                for arg in arg2:
                    #this is the only place that can change to read from DB insetad of dataframe..
                    tdf = df[(df.word == w) & (df.relns.str.contains(reln)) & (df.arg2.str.contains(arg))]
                    tls = []
                    cand_ids = [idd for idd,rl,rg in zip(list(tdf.document_id),list(tdf.relns),list(tdf.arg2)) \
                               if np.sum([1 for r,a in zip(rl.split(','),rg.split(',')) if r == reln and a == arg])>=1.0]
                    cand_ids = topic_doc_df[(topic_doc_df.document_ids.isin(cand_ids)) & \
                                           (topic_doc_df.topic_id == tid) & \
                                            (topic_doc_df.probability_score>=0.7)].document_ids.tolist()
                    if len(cand_ids)>0:
                        #print('topic -- {0}, word -- {1}, reln -- {2}, arg -- {3}, docs: {4}'.format(tid,w,reln,arg,cand_ids))
                        ls.append({'topic_id':tid,'word':w,\
                      'relns':reln,\
                      'arg2':arg, 'document_ids':cand_ids})
                

    res = pd.DataFrame(ls)
    res.to_csv('post_hoc_examples.csv')




if __name__ == "__main__":
    main()
