import re

days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def dateParser(date):
    # generate triple from a date in the form of integer, such as 20150101
    year = date / 10000
    month = date / 100 % 100
    day = date % 100
    return year, month, day

def dateEqual(y1, m1, d1, y2, m2, d2):
    # judge if two dates are the same
    if y1 != y2 or m1 != m2 or d1 != d2:
        return False
    return True

def leapYear(y):
    if y % 100 == 0:
        return y % 400 == 0
    return y % 4 == 0

def nextDate(y, m, d):
    topDay = days[m - 1] + (1 if (m == 2 and leapYear(y)) else 0)
    d += 1
    if d > topDay:
        d = 1
        m += 1
        if m > 12:
            m = 1
            y += 1
    return y, m, d

def dateToString(y, m, d):
    y, m, d = str(y), str(m), str(d)
    if len(m) == 1:
        m = '0' + m
    if len(d) == 1:
        d = '0' + d
    return y + m + d

def dateToFormalString(y, m, d):
    y, m, d = str(y), str(m), str(d)
    if len(m) == 1:
        m = '0' + m
    if len(d) == 1:
        d = '0' + d
    return y + '/' + m + '/' + d

def dateToInt(y, m, d):
    return y * 10000 + m * 100 + d