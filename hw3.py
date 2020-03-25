#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-

###################################################################################################
# @File main.py
# Main function of Gesture Recognition system
#
# @author Bin Hu <bh439>
# version: v1.0
# date: Mar-22-2020
# modefiy: Mar-25-2020
###################################################################################################

import numpy as np
import csv
import arrow

absolute_err_all = []
relative_err_all = []
class Const(object):
    ALAPH = 5e-3
    BETA = 11.1
    DEGREE = 7
    STOCK_DATA_PATH = './datasets'
    STORCK_DATA_FILES_LEN = 11
    STORCK_DATA_FILES_NAME = 'stock_data_'
    CSV_FILE_NAME = '.csv'

class BayesianCurvefitting(object):
    def read_stock_data(self, file):
        x_raw = []
        y_raw = []
        x_today = []
        y_today = []
        with open(file) as cf:
            reader = csv.DictReader(cf)
            row = reader.__next__()
            # get timestamp and the price of close today
            x_today.append(arrow.get(row['timestamp']).replace(tzinfo='US/Pacific').timestamp)
            y_today.append(float(row['close']))
            for row in reader:
                # get timestamp and the price of close every day before
                timestamp = arrow.get(row['timestamp']).replace(tzinfo='US/Pacific').timestamp
                x_raw.append(timestamp)
                y_raw.append(float(row['close']))
            # get the earliest timestamp
            time_earliest = arrow.get(row['timestamp']).replace(tzinfo='US/Pacific').timestamp
            # calculate the number of days from the earliest day
            x_raw = [int((i-time_earliest)/86400) for i in x_raw]
            x_today = [int((x_today[0]-time_earliest)/86400)]
        return np.asarray(x_raw), np.asarray(y_raw), np.asarray(x_today), np.asarray(y_today)

    def phi(self, x):
        phi = [[x**i] for i in range(Const.DEGREE + 1)]
        return np.asarray(phi)

    # formula (1.70), mean
    def mx(self, x, x_train, y_train, S):
        return Const.BETA*(self.phi(x).T).dot(S).dot(np.sum([t*self.phi(xt) for xt, t in zip(x_train, y_train)], axis=0))[0][0]

    # formula (1.71), variance
    def s2x(self, x, S):
        return (1/Const.BETA + (self.phi(x).T).dot(S.dot(self.phi(x))))[0][0]

    # formula (1.72)
    def S(self, x_train):
        S_inv = Const.ALAPH * np.identity(Const.DEGREE + 1) + Const.BETA * np.sum([self.phi(x).dot(self.phi(x).T) for x in x_train], axis=0)
        return np.linalg.inv(S_inv)

    def run(self):
        for i in range(Const.STORCK_DATA_FILES_LEN):
            stock_file = Const.STORCK_DATA_FILES_NAME + str(i)+ Const.CSV_FILE_NAME
            print("\nThe " + str(i+1)+ "th input stock dataset from the " + stock_file)

            stock_file = Const.STOCK_DATA_PATH + '/' + stock_file
            x_all, y_all, x_t, y_t = self.read_stock_data(stock_file)
            '''
            # x_train = x_all[int(len(x_all)/3):]
            # y_train = y_all[int(len(y_all)/3):]
            # x_test = x_all[:int(len(x_all)/3)]
            # y_test = y_all[:int(len(y_all)/3)]
            '''

            N = len(x_all)
            print("Data size: " + str(N+1))
            x_train = np.arange(0, 1.0, 1.0 / N)
            y_train = y_all[::-1]
            x_test = np.arange(0, 1.0 + 1.0 / N, 1.0 / N)

            S = self.S(x_train)

            predict_v = self.mx(x_test[-2],x_train, y_train,S)
            variance = self.s2x(x_test[-2], S)
            print("The prediction of N+1 time is", predict_v, "+-", variance)
            print("The real value is", y_t)

            absolute_err = abs(y_t-predict_v)
            relative_err = absolute_err/y_t
            absolute_err_all.append(absolute_err)
            relative_err_all.append(relative_err)

            print("\nThe absolute error is", absolute_err)
            print("The relative error is", relative_err)


        absolute_mean_err = np.average(absolute_err_all)
        ave_relative_err = np.average(relative_err_all)

        print("The overall absolute mean error is ", absolute_mean_err)
        print("The overall average relative error is ", ave_relative_err)

#######################################  Main Funciton   ##########################################
def main():
    # Show main menu
    myBFC = BayesianCurvefitting()
    myBFC.run()

###################################################################################################

if __name__ == "__main__":
    main()
