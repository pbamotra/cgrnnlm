#include "faster-rnnlm/context.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include "faster-rnnlm/layers/interface.h"
#include "faster-rnnlm/nce.h"
#include "faster-rnnlm/util.h"
#include "faster-rnnlm/words.h"



Context::Context( const ContextConfig &cfg ): cfg(cfg), word_to_lda_index(), beta_matrix(NULL), context_matrix(NULL) {
		Init();
}

void Context::Init() {
		ReadLDAVocab();
		ReadBetaMatrix();
}

void Context::ReadLDAVocab() {
    // TODO: dict_filepath should come from flag.
  std::fstream fin(cfg.dict_filepath.c_str());
  assert(fin != NULL);

  std::string word;
  unsigned long index;

  while(fin >> index  >> word) {
    word_to_lda_index[word] = index;
  }
}

void Context::ReadBetaMatrix() {
  std::ifstream indata;
  // beta_filepath is global. TODO
  indata.open(cfg.beta_filepath.c_str());
  std::string line;

  beta_matrix->resize(cfg.context_size,cfg.vocab_size);

  int i = 0;
  int j = 0;
  while (std::getline(indata, line))
  {
    std::stringstream          lineStream(line);
    std::string                cell;
    std::vector< double>       curr_row;
    j = 0;
    while (std::getline(lineStream, cell, ','))
    {
      (*beta_matrix)(i, j++) = atof(cell.c_str()); 
    }
    assert(j == cfg.vocab_size);
    ++i;
  }
  assert(i == cfg.context_size);
}

RowVector Context::get_beta_by_word(std::string word) {
  unsigned long n_beta_rows;
  unsigned long n_beta_cols;

  n_beta_rows = beta_matrix->rows();
  n_beta_cols = beta_matrix->cols();
  if (n_beta_rows > 0 && n_beta_cols > 0) {
    bool word_in_dict = word_to_lda_index.count(word) == 1;

    RowVector result;
    result.resize(1, n_beta_rows);
    result.setZero();

    if (word_in_dict) {
      unsigned long index = word_to_lda_index[word];
      if (n_beta_cols > index) {
        for(unsigned long row=0; row<n_beta_rows; row++) {
	  result(0, row) = (*beta_matrix)(row,index);
   	}
      }
    }
    return result;
  } else {
    fprintf(stderr, "Beta matrix is empty");
    std::exit(1);
  }
}


void Context::ComputeContextMatrix(Vocabulary vocab, const WordIndex *sen, const int seq_length) {
  if (context_matrix == NULL) {
    fprintf(stderr, "Provided a null context matrix. What did you expect?\n");
    return;
  }
	context_matrix->resize(seq_length, cfg.context_size);
  context_matrix->setZero();
  
	unsigned int sent_length = context_matrix->rows();
  unsigned int i=1;
  for (;i<sent_length; i++) {
    std::string curr_word(vocab.GetWordByIndex(sen[i]));
    context_matrix->row(i-1) = get_beta_by_word(curr_word).row(0);
  }
  context_matrix->row(i-1).setZero();
}


void Context::ComputeContextMatrix(Vocabulary vocab, const WordIndex *sen, RowMatrix * temp_context_matrix) {
  if (temp_context_matrix == NULL) {
    fprintf(stderr, "Provided a null context matrix. What did you expect?\n");
    return;
  }
  temp_context_matrix->setZero();
  
	unsigned int sent_length = temp_context_matrix->rows();
  unsigned int i=1;
  for (;i<sent_length; i++) {
    std::string curr_word(vocab.GetWordByIndex(sen[i]));
    temp_context_matrix->row(i-1) = get_beta_by_word(curr_word).row(0);
  }
  temp_context_matrix->row(i-1).setZero();
}

void Context::ComputeContextMatrixWithPrev(Vocabulary vocab, const WordIndex *sen,const int seq_length, int prev=2) {
  if (context_matrix == NULL) {
    fprintf(stderr, "Provided a null context matrix. What did you expect?\n");
    return;
  }
	context_matrix->resize(seq_length, cfg.context_size);
  context_matrix->setZero();
  int sent_length = context_matrix->rows();
  int i=1;

  RowMatrix temp_context_matrix(sent_length, context_matrix->cols());
	ComputeContextMatrix(vocab, sen, &temp_context_matrix); 

  for (; i<=sent_length; i++) {
        std::string curr_word(vocab.GetWordByIndex(sen[i]));
        context_matrix->row(i - 1) = get_beta_by_word(curr_word).row(0);
        for (int j=i-2; (j >= 0) && (j >= (i - 1 - prev)); j--) {
          context_matrix->row(i - 1) += temp_context_matrix.row(j);
        }
  }
}


void Context::ComputeContextMatrixAll(Vocabulary vocab, const WordIndex *sen, const int seq_length) {
		if (cfg.choice == 1)
				ComputeContextMatrix(vocab,sen,seq_length);
		else
				ComputeContextMatrixWithPrev(vocab,sen, seq_length);
}
