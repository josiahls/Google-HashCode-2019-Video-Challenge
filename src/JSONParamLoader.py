import os
from pandas.io.json import json
from pathlib import Path

class JSONParamLoader:

    def getParams(self):
        files = []
        abs_path = os.path.join(str(Path(__file__).parents[0]), 'logs')
        for file in os.listdir(abs_path):
            files.append(file)

        return json.read_json(os.path.join(abs_path, sorted(files)[-1]))

    def getBestParam(self):
        df = self.getParams()
        return df.loc[df['target'].argmax()]['params']

if __name__ == '__main__':
    json_data = JSONParamLoader().getBestParam()
