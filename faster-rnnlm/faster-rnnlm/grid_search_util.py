import os
import subprocess

SCRATCH_DIR = '/scratch/sbandiat/pgm-models/new'

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

    def _MakeDirs(self, log_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def _DeleteIfExists(self, file_name):
        if os.path.exists(file_name):
            print 'Deleting file %s' % file_name
            os.remove(file_name)

    def Run(self):
        cmd = './rnnlm %s' % ' '.join(self.flags)
        log_dir = 'logs'
        self._MakeDirs(log_dir)
        log_file = os.path.join(log_dir, '%s.log.txt' % self.GetModelName())
        self._DeleteIfExists(self.model_file)
        self._DeleteIfExists(self.model_file + '.nnet')
        _RunCmd(cmd.split(' '), log_file)
        test_entropy = self.GetTestEntropy(log_file)
        self._DeleteIfExists(self.model_file)
        self._DeleteIfExists(self.model_file + '.nnet')
        print 'Results: Model %s, Test Entropy: %s, Test Perplexity: %s' % (
                self.GetModelName(), test_entropy, 2**test_entropy) 

    def _AddPTBFlags(self):
        base_dir = '../benchmarks'
        data_dir = os.path.join(base_dir, 'simple-examples', 'data')
        train_file = os.path.join(data_dir, 'ptb.train.txt')
        validation_file = os.path.join(data_dir, 'ptb.valid.txt')
        self.model_file = os.path.join(SCRATCH_DIR, 'models', '%s_ptb' %  self.GetModelName())
        test_file = os.path.join(data_dir, 'ptb.test.txt')
        self._AddExecutionFlags(self.model_file, train_file, validation_file, test_file)

    def _AddExecutionFlags(self, model_file, train_file, validation_file, test_file):
        model_dir = os.path.dirname(model_file)
        self._MakeDirs(model_dir)
        self.AddFlag('train_and_test', 1)
        self.AddFlag('rnnlm', model_file)
        self.AddFlag('train', train_file)
        self.AddFlag('valid', validation_file)
        self.AddFlag('test', test_file)

    def GetModelName(self):
        m_name = ''
        for s_name, s_val in self.settings.iteritems():
            s_val = str(s_val)
            if '/' in s_val:
                continue
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
