'''
NTU nmlab project 2025
mock algorithm generation
'''
from generator import generate, write_files
from algorithm.runner import runner
if __name__ == '__main__':
    runner.run()
    links = ['L1','L4','L5']
    flow_hop = {"f1":"h1", "f2":"h3"}
    cmd, cfg = generate(links,flow_hop)
    write_files(cmd, cfg)