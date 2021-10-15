from pysrc.processor import TextClassifierProcessor
import rjieba
import re
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers import RoFormerConfig, RoFormerForSequenceClassification, RoFormerTokenizer
from pysrc.utils import InputExample
import math
MODEL_CLASSES = {
    "roformer": (RoFormerConfig, RoFormerForSequenceClassification, RoFormerTokenizer)
}

stopwords = [i.strip() for i in open('pysrc/cn_stopwords.txt').readlines()]
def pretty_cut(sentence):
    cut_list = rjieba.cut(''.join(re.findall('[\u4e00-\u9fa5]', sentence)))
    for i in range(len(cut_list)-1, -1, -1):
        if cut_list[i] in stopwords:
            del cut_list[i]
    return ''.join(cut_list)

def tensor_to_cpu(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("tensor type: expected one of (torch.Tensor)")
    return tensor.detach().cpu()

class CmccProcessor(TextClassifierProcessor):
    def get_labels(self):
        """See base class."""
        return np.arange(0,109,1).astype(str).tolist()

    def read_data(self, input_file):
        """Reads a json list file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            text_b = None
            label = line[0]
            examples.append(
                InputExample(guid=guid, texts=[text_a, text_b], label=label)
            )
        return examples

ays = [('婚姻家庭纠纷','1'),('婚约财产纠纷','2'),('婚内夫妻财产分割纠纷','2'),('离婚纠纷','2'),('离婚后财产纠纷','2'),('离婚后损害责任纠纷','2'),('婚姻无效纠纷','2'),('撤销婚姻纠纷','2'),('夫妻财产约定纠纷','2'),('同居关系纠纷','2'),('同居关系析产纠纷','3'),('同居关系子女抚养纠纷','3'),('亲子关系纠纷','2'),('确认亲子关系纠纷','3'),('否认亲子关系纠纷','3'),('抚养纠纷','2'),('抚养费纠纷','3'),('变更抚养关系纠纷','3'),('扶养纠纷','2'),('扶养费纠纷','3'),('变更扶养关系纠纷','3'),('赡养纠纷','2'),('赡养费纠纷','3'),('变更赡养关系纠纷','3'),('收养关系纠纷','2'),('确认收养关系纠纷','3'),('解除收养关系纠纷','3'),('监护权纠纷','2'),('探望权纠纷','2'),('分家析产纠纷','2'),('合同纠纷','1'),('房屋买卖合同纠纷','2'),('商品房预约合同纠纷','3'),('商品房预售合同纠纷','3'),('商品房销售合同纠纷','3'),('商品房委托代理销售合同纠纷','3'),('经济适用房转让合同纠纷','3'),('农村房屋买卖合同纠纷','3'),('民事主体间房屋拆迁补偿合同纠纷','2'),('借款合同纠纷','2'),('金融借款合同纠纷','3'),('同业拆借纠纷','3'),('民间借贷纠纷','3'),('小额借款合同纠纷','3'),('金融不良债权转让合同纠纷','3'),('金融不良债权追偿纠纷','3'),('保证合同纠纷','2'),('抵押合同纠纷','2'),('质押合同纠纷','2'),('定金合同纠纷','2'),('租赁合同纠纷','2'),('土地租赁合同纠纷','3'),('房屋租赁合同纠纷','3'),('车辆租赁合同纠纷','3'),('建筑设备租赁合同纠纷','3'),('融资租赁合同纠纷','2'),('劳务合同纠纷','2'),('离退休人员返聘合同纠纷','2'),('知识产权权属、侵权纠纷','1'),('商标权权属、侵权纠纷','2'),('商标权权属纠纷','3'),('侵害商标权纠纷','3'),('专利权权属、侵权纠纷','2'),('专利申请权权属纠纷','3'),('专利权权属纠纷','3'),('侵害发明专利权纠纷','3'),('侵害实用新型专利权纠纷','3'),('侵害外观设计专利权纠纷','3'),('假冒他人专利纠纷','3'),('发明专利临时保护期使用费纠纷','3'),('职务发明创造发明人、设计人奖励、报酬纠纷','3'),('发明创造发明人、设计人署名权纠纷','3'),('劳动争议','1'),('劳动合同纠纷','2'),('确认劳动关系纠纷','3'),('集体合同纠纷','3'),('劳务派遣合同纠纷','3'),('非全日制用工纠纷','3'),('追索劳动报酬纠纷','3'),('经济补偿金纠纷','3'),('竞业限制纠纷','3'),('社会保险纠纷','2'),('养老保险待遇纠纷','3'),('工伤保险待遇纠纷','3'),('医疗保险待遇纠纷','3'),('生育保险待遇纠纷','3'),('失业保险待遇纠纷','3'),('保险纠纷','1'),('财产保险合同纠纷','2'),('财产损失保险合同纠纷','3'),('责任保险合同纠纷','3'),('信用保险合同纠纷','3'),('保证保险合同纠纷','3'),('保险人代位求偿权纠纷','3'),('人身保险合同纠纷','2'),('人寿保险合同纠纷','3'),('意外伤害保险合同纠纷','3'),('健康保险合同纠纷','3'),('再保险合同纠纷','2'),('保险经纪合同纠纷','2'),('保险代理合同纠纷','2'),('进出口信用保险合同纠纷','2'),('保险费纠纷','2'),('侵权责任纠纷','1'),('机动车交通事故责任纠纷','2'),('非机动车交通事故责任纠纷','2'),('医疗损害责任纠纷','2'),('侵害患者知情同意权责任纠纷','3'),('医疗产品责任纠纷','3')]

class Predictor:
    def __init__(self):
        self.model_type = 'roformer'
        self.num_labels = 109
        self.model_path = f'./pysrc/pretrained/{self.model_type}/'
        self.device = 'cuda:0'
        self.eval_max_seq_length = 1500
        self.per_gpu_eval_batch_size = 64
        self.init_weight()
        
    def init_weight(self):
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.model_type]
        self.tokenizer = tokenizer_class.from_pretrained(
            self.model_path, do_lower_case=True
        )
        self.processor = CmccProcessor(
            data_dir=None, tokenizer=self.tokenizer, prefix=None
        )
        config = config_class.from_pretrained(
            self.model_path,
            num_labels=109,
            cache_dir=None,
        )
        self.model = model_class.from_pretrained(self.model_path, config=config)
        
    

    def build_eval_dataloader(self, eval_dataset):
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        batch_size = self.per_gpu_eval_batch_size
        sampler = SequentialSampler(eval_dataset)
        data_loader = DataLoader(eval_dataset, sampler=sampler, batch_size=batch_size, 
                                 collate_fn=self.processor.collate_fn,num_workers=4)
        return data_loader
    
    def build_inputs(self, batch):
        '''
        Sent all model inputs to the appropriate device (GPU on CPU)
        rreturn:
         The inputs are in a dictionary format
        '''
        batch = tuple(t.to(self.device) for t in batch)
        inputs = {key: value for key, value in zip(self.processor.get_input_keys(), batch)}
        return inputs
    
    def predict(self, records):
        indexs = []
        ran = 640
        for i in range(0,math.floor(len(records)/ran)+1):
            vo = records[i*ran:(i+1)*ran]
            vo = [pretty_cut(x) for x in vo]
            result = []
            eval_dataset = self.processor.create_dataset(
                self.eval_max_seq_length, vo, "test"
            )
            eval_dataloader = self.build_eval_dataloader(eval_dataset)
            for step, batch in enumerate(eval_dataloader):
                self.model.eval()
                inputs = self.build_inputs(batch)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                logits = outputs[1]
                result.append(tensor_to_cpu(logits))
            result = torch.cat(result, dim=0)
            topk = max((1,))
            _, pred = result.topk(topk, 1, True, True)
            top_index = pred.cpu().numpy()
            for x in top_index:
                indexs.append({
                        "aydj":ays[x[0]][1],
                        "ay":ays[x[0]][0]
                    })
        return indexs