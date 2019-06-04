import logging
import pandas as pd

from naruto_skills import solr


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # From 2 domains: TV and milk
    topics = {'4489', '4637', '5200', '6732', '5651', '4084', '3245', '2638', '23709', '23708', '21357'}
    start = '2018-12-01T00:00:00'
    end = '2019-05-01T00:00:00'
    filters = (
        'q=*:*',
        'fq=-is_ignore:1',
        'fq=-is_noisy:1',
        'fq=is_approved:1',
        'wt=json',
        'fq=search_text:*',
        'fq=copied_at:[%sZ TO %sZ]' % (start, end)
    )
    list_df_datas = [solr.crawl_topic(domain='http://solrtopic.younetmedia.com', topic=topic, filters=filters,
                                      limit=100000, batch_size=5000, username='trind', password='Jhjhsdf$3&sdsd')
                     for topic in topics]

    df = pd.concat(list_df_datas)
    df['id'] = df['id'].map(str)
    df.drop_duplicates(subset=['id'])
    df.dropna(how='all')

    path_to_save = '/source/main/data_download/output/raw_data.csv'
    logging.info('There are totally %s records', df.shape[0])
    logging.info('Saved all data of domain TV to %s', path_to_save)
    df.to_csv(path_to_save, index=None)

