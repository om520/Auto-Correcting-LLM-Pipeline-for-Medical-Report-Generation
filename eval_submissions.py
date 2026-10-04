import pandas as pd
import textwrap

try:
    test = pd.read_csv(r'C:\Users\Om mishra\Downloads\test.csv')
    sub_new = pd.read_csv(r'C:\Users\Om mishra\Downloads\intermediate_submission (1).csv')
    sub_old = pd.read_csv(r'C:\Users\Om mishra\Desktop\radiology_llm\nxtwave_om_mishra_tf_ultra9.csv')
    
    # Merge all
    df = test[['case_id', 'template_content', 'dictation']].merge(
        sub_new.rename(columns={'report': 'report_new'}), on='case_id'
    ).merge(
        sub_old.rename(columns={'report': 'report_old'}), on='case_id'
    )
    
    def jaccard(s1, s2):
        set1, set2 = set(str(s1).lower().split()), set(str(s2).lower().split())
        if not set1 or not set2: return 0.0
        return len(set1 & set2) / len(set1 | set2)
        
    df['sim_new'] = df.apply(lambda r: jaccard(r['template_content'], r['report_new']), axis=1)
    df['sim_old'] = df.apply(lambda r: jaccard(r['template_content'], r['report_old']), axis=1)
    
    print('--- TEMPLATE PRESERVATION SCORE (Higher is better) ---')
    print(f'New (Qwen 7B Pipeline): {df["sim_new"].mean():.4f}')
    print(f'Old (Llama 8B script):  {df["sim_old"].mean():.4f}')
    print('\n* Note: Higher preservation means fewer penalties for changing normal text.\n')
    
    print('--- CASE COMPARISON (Sample) ---')
    sample = df.iloc[0]
    print('DICTATION:')
    print(textwrap.indent(str(sample['dictation']), '  '))
    print('-'*50)
    print('OLD REPORT (Llama 8B):')
    print(textwrap.indent(str(sample['report_old']).split('IMPRESSION:')[0][:300] + '...', '  '))
    print('-'*50)
    print('NEW REPORT (Qwen 7B):')
    print(textwrap.indent(str(sample['report_new']).split('IMPRESSION:')[0][:300] + '...', '  '))

except Exception as e:
    print('Error:', e)
