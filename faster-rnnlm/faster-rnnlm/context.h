#ifndef FASTER_RNNLM_CONTEXT_H_
#define FASTER_RNNLM_CONTEXT_H_
#include <inttypes.h>
#include <stdio.h>

#include <string>
#include <map>

#include "faster-rnnlm/hierarchical_softmax.h"
#include "faster-rnnlm/maxent.h"
#include "faster-rnnlm/settings.h"
#include "faster-rnnlm/util.h"

typedef std::map<std::string, unsigned long> WordToLDAIndex;
// class NNet;
struct ContextConfig {
  int context_size;
  int vocab_size;
  int choice;
  std::string dict_filepath;
  std::string beta_filepath;
};

class Context {
public:
  Context(const ContextConfig &cfg);
  ~Context();

  void ComputeContextMatrixAll(Vocabulary vocab, const WordIndex *sen,
                               const int seq_length);
  void ComputeContextMatrixWithPrev(Vocabulary vocab, const WordIndex *sen,
                                    const int seq_length, int prev);
  void ComputeContextMatrix(Vocabulary vocab, const WordIndex *sen,
                            const int seq_length);
  void ComputeContextMatrix(Vocabulary vocab, const WordIndex *sen,
                            RowMatrix *temp_context_matrix);
  RowVector get_beta_by_word(std::string word);

  WordToLDAIndex word_to_lda_index;
  RowMatrix *beta_matrix;
  RowMatrix *context_matrix;
  const ContextConfig cfg;

private:
  void Init();
  void ReadLDAVocab();
  void ReadBetaMatrix();
};

#endif  // FASTER_RNNLM_CONTEXT_H_
