#!/usr/bin/python
# -*- coding: utf-8 -*-
import io
import os
import re

import pandas as pd

from core.processing.stemmer import Stemmer


class Evaluator:

    def __init__(self, syn_dict_filepath, test_filepath, etalon_filepath):
        self.syn_dict = self._syn_dict_from_file(syn_dict_filepath)
        self.user_answers = test_filepath
        self.etalon_answers = etalon_filepath
        self.stemmer = Stemmer()


    @staticmethod
    def _syn_dict_from_file(filepath):
        syn_dict = []
        with io.open(filepath, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                args = line.split(',')
                syn_dict = [a.strip() for a in args]
        return syn_dict

    """ Проверка имен на заменяемость. Имена могут быть:
        - подстрокой другого: Путин vs Владимир Путин, Валентин vs Валентино
        - формой слова: соединенные vs соединенный vs соединить.
        "Словарь синонимов" хранится в переменной syn_dict.
    """
    def _checkWords(self, word1, word2):
        word1 = word1.decode('cp1251')
        word2 = word2.decode('utf-8')
        assert(type(word1) == unicode)
        assert(type(word2) == unicode)
        if word1 == word2:
            return True

        res2 = re.split(" ", word2)
        for r in res2:
            if word1 in r:
                return True
            for s in self.syn_dict:
                if word1 in s and r in s:
                    return True
            l = self.stemmer.lemmatize_to_str(r)
            if word1 in l:
                return True
            for s in self.syn_dict:
                if word1 in s and l in s:
                    return True

        res2 = re.split(" ", word1)
        for r in res2:
            if word2 in r:
                return True
            for s in self.syn_dict:
                if word2 in s and r in s:
                    return True
            l = self.stemmer.lemmatize_to_str(r)
            if word2 in l:
                return True
            for s in self.syn_dict:
                if word2 in s and l in s:
                    return True

        return False

    def _calcPrecisionAndRecall(self, results):
        """ Расчет полноты и точности.
        """
        # Берем все позитивные и негативные ответы команд
        pos_answers = results[(results['how_results'] == 'pos')]
        neg_answers = results[(results['how_results'] == 'neg')]

        # Расчет точности.
        if len(pos_answers) != 0:
            # print "-- {}".format(len(pos_answers[(pos_answers['comparison']==True)]))
            # print "-- {}".format(len(pos_answers))
            pos_prec = 1.0 * len(pos_answers[(pos_answers['comparison'] == True)]) / len(pos_answers)
            # print "== {}".format(pos_prec)
        else:
            pos_prec = 0.0
        if len(neg_answers) != 0:
            neg_prec = 1.0 * len(neg_answers[(neg_answers['comparison'] == True)]) / len(neg_answers)
        else:
            neg_prec = 0.0

        # Расчет полноты.
        if len(results[results['how_orig'] == 'pos']) != 0:
            pos_recall = 1.0 * len(pos_answers[(pos_answers['comparison'] == True)]) / len(results[results['how_orig'] == 'pos'])
        else:
            pos_recall = 0.0
        if len(results[results['how_orig'] == 'neg']) != 0:
            neg_recall = 1.0 * len(neg_answers[(neg_answers['comparison'] == True)]) / len(results[results['how_orig'] == 'neg'])
        else:
            neg_recall = 0.0

        assert(type(pos_prec) == float)
        assert(type(neg_prec) == float)
        assert(type(pos_recall) == float)
        assert(type(neg_recall) == float)

        return pos_prec, neg_prec, pos_recall, neg_recall

    def _calc_a_file(self, num):
        """ Data calculation for a file of 'num' index
        """
        # Если файл существует.
        filename = "{}/art{}.opin.txt".format(self.user_answers, str(num))
        if not os.path.exists(filename):
            print "missed: art{}.opin.txt".format(num)
            return 0, 0, 0, 0

        if os.stat(filename).st_size == 0:
            print "empty file: art{}.opin.txt".format(num)
            return 0, 0, 0, 0

        # Считываем файл ответов команды и отбрасываем лишнюю информацию.
        # print("test_"+color+"/art"+str(num)+".opin.txt", end=" ")


        f = pd.read_csv(filename, sep=',', header=None)
        # print(" read")
        orig_file = f[[0, 1, 2]].copy()
        orig_file.columns = ['who', 'to', 'how_orig']
        orig_file['who'] = orig_file['who'].str.strip()
        orig_file['who'] = orig_file['who'].str.lower()
        orig_file['to'] = orig_file['to'].str.strip()
        orig_file['to'] = orig_file['to'].str.lower()
        orig_file['how_orig'] = orig_file['how_orig'].str.strip()

        # Считываем файл ответов экспертов.
        file_experts = self.etalon_answers + "/art{}.opin.txt".format(str(num))
        print "reading: {}".format(file_experts)
        file2 = pd.read_csv(file_experts, sep=',', header=None)
        test_file = file2[[0, 1, 2]].copy()
        test_file.columns = ['who', 'to', 'how_results']
        test_file['who'] = test_file['who'].str.strip()
        test_file['who'] = test_file['who'].str.lower()
        test_file['to'] = test_file['to'].str.strip()
        test_file['to'] = test_file['to'].str.lower()
        test_file['how_results'] = test_file['how_results'].str.strip()

        orig_file = orig_file.sort_values(['who', 'to'])
        test_file = test_file.sort_values(['who', 'to'])
        # Сливаем ответы там, где имена собвпадают.
        results = test_file.merge(orig_file, 'outer', on=['who', 'to'], copy=False)
        results.insert(len(results.columns), 'comparison', '')
        # Сравниваем для них ответы
        results['comparison'] = results['how_results'] == results['how_orig']
        results = results.sort_values(['comparison', 'how_orig', 'how_results'])
        # Берем те части ответов, в которых имена не совпадают.
        faulty = results[results.comparison == False]
        count_res = len(faulty) - len(faulty[faulty.how_orig.isnull()])

        # Идем по всем ответам, сравниваем всех со всеми, может быть имя было
        # выделено командой не так, или эксперт написал синоним.
        has_changes = False
        for i in range(count_res, len(faulty)):
            for j in range(count_res):
                if(self._checkWords(results['who'][results.index[i]], results['who'][results.index[j]]) and
                   self._checkWords(results['to'][results.index[i]], results['to'][results.index[j]])):
                    # Если надо длеать замену, дописываем ответ в одну строчку и помечаем на удаление другую.
                    # print('<', results['who'][results.index[i]], '>,<', results['who'][results.index[j]], '>')
                    # print('<', results['to'][results.index[i]], '>,<', results['to'][results.index[j]], '>')
                    results.loc[results.index[i], ('how_orig')] = results.loc[results.index[j], ('how_orig')]
                    results.loc[results.index[i], ('comparison')] = (results.loc[results.index[i], ('how_orig')] == results.loc[results.index[i], ('how_results')])
                    results.loc[results.index[j], ('comparison')] = '---'
                    has_changes = True

        # Выкидываем все строки, которые были слиты вместе с другими.
        if has_changes:
            results = results[results['comparison'] != '---']

        # Сохраняем файл со сравнениями.
        comparison_file = self.user_answers + "/art{}.comp.txt".format(str(num))
        results.to_csv(comparison_file)

        return self._calcPrecisionAndRecall(results)

    def evaluate(self, test_indices):
        """ Main evaluation subprogram
        """
        pos_prec, neg_prec, pos_recall, neg_recall = (0, 0, 0, 0)

        for n in test_indices:
            [pos_prec1, neg_prec1, pos_recall1, neg_recall1] = self._calc_a_file(n)

            pos_prec += pos_prec1
            neg_prec += neg_prec1
            pos_recall += pos_recall1
            neg_recall += neg_recall1

        pos_prec /= len(test_indices)
        neg_prec /= len(test_indices)
        pos_recall /= len(test_indices)
        neg_recall /= len(test_indices)

        if pos_prec * pos_recall != 0:
            f1_pos = 2 * pos_prec * pos_recall / (pos_prec + pos_recall)
        else:
            f1_pos = 0

        if neg_prec * neg_recall != 0:
            f1_neg = 2 * neg_prec * neg_recall / (neg_prec + neg_recall)
        else:
            f1_neg = 0

        return {"pos_prec": pos_prec,
                "neg_prec": neg_prec,
                "pos_recall": pos_recall,
                "neg_recall": neg_recall,
                "f1_pos": f1_pos,
                "f1_neg": f1_neg}
