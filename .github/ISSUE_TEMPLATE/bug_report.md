---
name: バグレポート
about: 発生したバグの報告と修正依頼
タイトル: バグレポートのタイトル
---

**バグの概要**
ここにバグの概要を書く．

**バグが発生する条件**
バグを再現するための方法．
1. 〇〇をインポートする
2. ✕✕する
3. △△する
4. ...


バグが発生したときのコード:
```Python
import numpy

x = np.array([1, 2, 3])
print(x)
print('example of code')
```

**エラー出力**
```
example
with open('not_exist_file.txt') as f:
    print(f.read())

FileNotFoundError: [Errno 2] No such file or directory: 'not_exist_file.txt'
```

**その他**
他に何かあればここに書く．
