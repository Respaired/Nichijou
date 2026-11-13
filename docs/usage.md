# Usage and tips

1. You can generate up to 60 seconds of audio. If you use an audio prompt, its duration counts toward that limit, leaving less time for generated audio.

2. You can switch speakers using `<S1>` and `<S2>`, or specify their gender with `<Sn:male>` and `<Sn:female>`. However, `<S3>` and beyond were relatively rare in training, so they may not work as reliably.

3. Try to write your text the way a real conversation would flow. Include natural hesitations like "um" and "uh," false starts, self-corrections, and other disfluencies to make the speech sound more human. You can also check the default examples for inspiration.

4. Other supported control tags include:
   - `<laughs>`: Laughter
   - `<coughs>`: Coughing
   - `<clears_throat>`: Throat clearing
   - `<bgm>`: Background music
   - `<anime>`: Anime-style speech
   - `<en>`: English
   - `<ja>`: Japanese

5. It's recommended to prepend `<ja><S1>` to the beginning of each sample. However, **don't add these tags when using an audio prompt**.

6. Wrap English text in square brackets `[]`, like `[hello text]`, unless the entire text is in English. Check the examples for the correct formatting.

7. The `<anime>` tag currently doesn't support multi-speaker conversations.


___


**使い方とポイント**

1. 生成できる音声は最大60秒。音声プロンプトを使う場合は、その長さも60秒に含まれるため、生成できる時間がその分短くなります。

2. 話者の切り替えには `<S1>` と `<S2>` を使い、`<Sn:male>` や `<Sn:female>` で性別も指定できます。ただし、`<S3>` 以降は使用例が少なく、まだ動作が不安定な場合もあります。

3. テキストは、実際の会話の流れを意識して書くのがポイント。「えっと」「あの」といった言いよどみや、途中で言い直すような表現を混ぜると、より自然な会話になります。デフォルトのテキストも参考にしてみてください。

4. ほかにも、以下のタグが使えます。
   - `<laughs>`：笑い声
   - `<coughs>`：咳
   - `<clears_throat>`：咳払い
   - `<bgm>`：BGM
   - `<anime>`：アニメ風
   - `<en>`：英語
   - `<ja>`：日本語

5. 基本的には、各サンプルの冒頭に `<ja><S1>` を付けるのがおすすめです。ただし、音声プロンプトを使う場合は付けないでください。

6. 英語の部分は、`[hello text]` のように角括弧 `[]` で囲んでください。テキスト全体が英語の場合はその必要はありません。具体的な書き方はサンプルを参考にしてみてください。

7. `<anime>` タグは、複数人での会話には対応していません。