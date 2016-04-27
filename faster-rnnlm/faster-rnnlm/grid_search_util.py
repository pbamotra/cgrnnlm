import os
import subprocess

class Configuration(object):

    def __init__(self, **kwargs):
        self.flags = []
        self.settings = kwargs
        for flag_name, flag_val in kwargs.items():
            self.AddFlag(flag_name, flag_val)
        self._AddPTBFlags()

    def AddFlag(self, flag_name, val):
        if val is not None:
            #print 'Adding flag: -%s %s' % (flag_name, val)
            self.flags.append('--%s %s' % (flag_name, val))

    def Run(self):
        cmd = './rnnlm %s' % ' '.join(self.flags)
        log_file = '%s.log.txt' % self.GetModelName()
        _RunCmd(cmd.split(' '), log_file)
        test_entropy = self.GetTestEntropy(log_file)
        print 'Results: Model %s, Test Entropy: %s' % (self.GetModelName(), test_entropy) 

    def _AddPTBFlags(self):
        base_dir = '../benchmarks'
        data_dir = os.path.join(base_dir, 'simple-examples', 'data')
        train_file = os.path.join(data_dir, 'ptb.train.txt')
        validation_file = os.path.join(data_dir, 'ptb.valid.txt')
        model_file = os.path.join(base_dir, 'models', '%s_ptb' %  self.GetModelName())
        test_file = os.path.join(data_dir, 'ptb.test.txt')
        self._AddExecutionFlags(model_file, train_file, validation_file, test_file)

    def _AddExecutionFlags(self, model_file, train_file, validation_file, test_file):
        model_dir = os.path.dirname(model_file)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.AddFlag('train_and_test', 1)
        self.AddFlag('rnnlm', model_file)
        self.AddFlag('train', train_file)
        self.AddFlag('valid', validation_file)
        self.AddFlag('test', test_file)

    def GetModelName(self):
        m_name = ''
        for s_name, s_val in self.settings.iteritems():
            m_name += '%s_%s' % (s_name, s_val)
        return m_name

    def GetTestEntropy(self, log_file):
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('Test entropy'):
                    parts = line.split(' ')
                    return float(parts[2])
        return None


def _RunCmd(cmd_list, log_file):
    with open(log_file, 'w') as f:
        p = subprocess.Popen(cmd_list, stdout=f, stderr=f)
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            print '*' * 50
            print '%s Failed!' % cmd_list
            print 'stdout:\n %s, stderr: %s\n' % (stdout, stderr)
            raise Exception('Failed to run command')
        else:
            print 'Job %s successful' % cmd_list

def Compile():
    _RunCmd(['make', '-j5'], 'compiler_result.txt')
