#include "faster-rnnlm/context.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include "faster-rnnlm/layers/interface.h"
#include "faster-rnnlm/nce.h"
#include "faster-rnnlm/util.h"
#include "faster-rnnlm/words.h"

Context::Context(const ContextConfig &cfg)
    : cfg(cfg), word_to_lda_index(), lda_vocab_size(0) {
  Init();
}

void Context::Init() {
  fprintf(stderr, "debug-line-1\n");
  ReadLDAVocab();
  fprintf(stderr, "debug-line-2\n");
  ReadBetaMatrix();
  fprintf(stderr, "debug-line-3\n");
}

void Context::ReadLDAVocab() {
  // TODO: dict_filepath should come from flag.
  std::fstream fin(cfg.dict_filepath.c_str());
  if (fin == NULL) {
    fprintf(stderr, "Can't read file %s", cfg.dict_filepath.c_str());
  }
  assert(fin != NULL);

  std::string word;
  unsigned long index;

  while (fin >> index >> word) {
    word_to_lda_index[word] = index;
  }
  lda_vocab_size = word_to_lda_index.size();
}

void Context::ReadBetaMatrix() {
  fprintf(stderr, "debug-beta-line-1\n");
  std::ifstream indata;
  indata.open(cfg.beta_filepath.c_str());
  std::string line;

  fprintf(stderr, "debug-beta-line-2, c1 = %d, v_size=%d\n", cfg.context_size,
          lda_vocab_size);
  beta_matrix.resize(cfg.context_size, lda_vocab_size);

  fprintf(stderr, "debug-beta-line-3\n");
  int i = 0;
  int j = 0;
  while (std::getline(indata, line)) {
    std::stringstream lineStream(line);
    std::string cell;
    std::vector<double> curr_row;
    j = 0;
    while (std::getline(lineStream, cell, ',')) {
      beta_matrix(i, j++) = atof(cell.c_str());
    }
    assert(j == lda_vocab_size);
    ++i;
  }
  fprintf(stderr, "debug-beta-line-4\n");
  assert(i == cfg.context_size);
}

RowVector Context::get_beta_by_word(std::string word) {
  unsigned long n_beta_rows;
  unsigned long n_beta_cols;

  n_beta_rows = beta_matrix.rows();
  n_beta_cols = beta_matrix.cols();
  if (n_beta_rows > 0 && n_beta_cols > 0) {
    bool word_in_dict = word_to_lda_index.count(word) == 1;

    RowVector result;
    result.resize(1, n_beta_rows);
    result.setZero();

    if (word_in_dict) {
      unsigned long index = word_to_lda_index[word];
      if (n_beta_cols > index) {
        for (unsigned long row = 0; row < n_beta_rows; row++) {
          result(0, row) = beta_matrix(row, index);
        }
      }
    }
    return result;
  } else {
    fprintf(stderr, "Beta matrix is empty");
    std::exit(1);
  }
}

void Context::ComputeContextMatrix(Vocabulary vocab, const WordIndex *sen,
                                   const int seq_length,
                                   RowMatrix *context_matrix) {
  if (context_matrix == NULL) {
    fprintf(stderr, "Provided a null context matrix. What did you expect?\n");
    return;
  }
  fprintf(stderr, "Trying to resize cm to (%d, %d)\n", seq_length, cfg.context_size);
  context_matrix->resize(seq_length, cfg.context_size);
  context_matrix->setZero();

  fprintf(stderr, "Computing cm..\n");
  for (int i = 0; i < seq_length; i++) {
    std::string curr_word(vocab.GetWordByIndex(sen[i]));
    context_matrix->row(i) = get_beta_by_word(curr_word).row(0);
  }
  fprintf(stderr, "Successfully ocmputec cm\n");
}

void Context::ComputeContextMatrixWithPrev(Vocabulary vocab,
                                           const WordIndex *sen,
                                           const int seq_length,
                                           RowMatrix *context_matrix) {
  if (context_matrix == NULL) {
    fprintf(stderr, "Provided a null context matrix. What did you expect?\n");
    return;
  }
  ComputeContextMatrix(vocab, sen, seq_length, context_matrix);
}

void Context::ComputeContextMatrixAll(Vocabulary vocab, const WordIndex *sen,
                                      const int seq_length,
                                      RowMatrix *context_matrix) {
  if (cfg.choice == 1)
    ComputeContextMatrix(vocab, sen, seq_length, context_matrix);
  else
    ComputeContextMatrixWithPrev(vocab, sen, seq_length, context_matrix);
}
