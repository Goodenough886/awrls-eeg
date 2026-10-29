"""Reassemble ordered gzip CSV parts without changing the published data."""
from pathlib import Path
import csv,gzip,json

def main():
    root=Path(__file__).resolve().parent
    for table in json.loads((root/'published_results/TABLE_INDEX.json').read_text(encoding='utf-8')):
        dest=root/'local_runs/restored_tables'/table['source_table']
        if dest.exists():raise FileExistsError(dest)
        dest.parent.mkdir(parents=True,exist_ok=True)
        count=0;header=None
        with dest.open('w',encoding='utf-8',newline='') as out:
            writer=csv.writer(out,lineterminator='\n')
            for part in table['parts']:
                with gzip.open(root/part['path'],'rt',encoding='utf-8',newline='') as f:
                    reader=csv.reader(f);h=next(reader)
                    if header is None:header=h;writer.writerow(h)
                    elif h!=header:raise ValueError('Part header mismatch')
                    for row in reader:writer.writerow(row);count+=1
        if count!=table['rows']:raise ValueError('Row count mismatch')
        print(table['source_table'],count)

if __name__=='__main__':main()
