# Trigger Points
Our online and offline evaluations both consider trigger points:
symbols or keywords that automatically trigger code completion.

We used the following trigger points:
```
await assert raise del lambda yield return while for
if elif else global in and not or is with except
. + - * / % ** << >> & | ^ == != <= >= += -= = < > ;
, [ ( { ~ 
```

## Online Evaluation - Trigger Point Distribution
The following table highlights how many datapoints we have for
different trigger points in the online evaluation.

| **Trigger Point** | **Frequency** |
|-------------------|---------------|
| `.`               | 180136        |
 | `(`               | 150470        |
 | `=`               | 137863        |
 | `,`               | 63383         |
 | `/`               | 44124         |
 | `-`               | 42826         |
 | `[`               | 29372         |
 | `<`               | 27915         |
 | `{`               | 20312         |
 | `if`              | 19767         |
 | `>`               | 15653         |
 | `return`          | 15244         |
 | `*`               | 15244         |
 | `+`               | 15218         |
 | `in`              | 12655         |
 | `or`              | 9255          |
 | `==`              | 5727          |
 | `;`               | 5674          |
 | `and`             | 5336          |
 | `is`              | 5024          |
 | `&`               | 4719          |
 | `for`             | 4319          |
 | `%`               | 4268          |
 | `\|`              | 3828          |
 | `else`            | 3205          |
 | `not`             | 2393          |
 | `with`            | 2310          |
 | `while`           | 1290          |
 | `await`           | 1177          |
 | `!=`              | 1073          |
 | `+=`              | 963           |
 | `del`             | 794           |
 | `**`              | 709           |
 | `^`               | 706           |
 | `assert`          | 649           |
 | `~`               | 546           |
 | `except`          | 496           |
 | `>=`              | 423           |
 | `>>`              | 414           |
 | `<<`              | 404           |
 | `global`          | 372           |
 | `raise`           | 351           |
 | `<=`              | 344           |
 | `elif`            | 248           |
 | `yield`           | 215           |
 | `-=`              | 152           |
 | `lambda`          | 118           |
