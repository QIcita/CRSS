"""Check training protocol identity after normalizing candidate labels only."""
import ast
from .common import ROOT,PARENT,save


class Normalize(ast.NodeTransformer):
    def visit_Constant(self,node):
        if node.value=='r38':node.value='r34'
        return node
    def visit_Tuple(self,node):
        node=self.generic_visit(node)
        if all(isinstance(x,ast.Constant) and isinstance(x.value,str) for x in node.elts):
            unique=[]
            for x in node.elts:
                if not any(y.value==x.value for y in unique):unique.append(x)
            node.elts=unique
        return node


def main():
    parent=ast.parse((PARENT/'tools/cpsf_r34_fps_rebench/worker.py').read_text())
    own=Normalize().visit(ast.parse((ROOT/'tools/cpsf_cmm_pair_fusion/worker.py').read_text()))
    functions={}
    for name in ('state_steps','order_for','optimizer','metrics','evaluate','train'):
        a=next(x for x in parent.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==name)
        b=next(x for x in own.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==name)
        functions[name]=ast.dump(a)==ast.dump(b)
    assert all(functions.values()),functions
    report=dict(passed=True,identity_normalization='r38 to r34; duplicate exclusion labels removed',functions=functions)
    save(ROOT/'worker_protocol_audit.json',report);print(report)


if __name__=='__main__':main()
