# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import glob
import io
import logging
import unicodedata
from pathlib import Path, PurePath
from typing import Callable, Optional, Union

import lmdb
from PIL import Image,ImageFile
from torch.utils.data import Dataset, ConcatDataset

from strhub.data.utils import CharsetAdapter
import torch
import numpy as np
import re
import cv2 as cv
import os
log = logging.getLogger(__name__)


def build_tree_dataset(root: Union[PurePath, str], *args, **kwargs):
    try:
        kwargs.pop('root')  # prevent 'root' from being passed via kwargs
    except KeyError:
        pass
    root = Path(root).absolute()
    log.info(f'dataset root:\t{root}')
    datasets = []
    for mdb in glob.glob(str(root / '**/data.mdb'), recursive=True):
        mdb = Path(mdb)

        ds_name = str(mdb.parent.relative_to(root))
        ds_root = str(mdb.parent.absolute())
        dataset = LmdbDataset(ds_root, *args, **kwargs)
        log.info(f'\tlmdb:\t{ds_name}\tnum samples: {len(dataset)}')
        datasets.append(dataset)
    return ConcatDataset(datasets)


class LmdbDataset(Dataset):
    """Dataset interface to an LMDB database.

    It supports both labelled and unlabelled datasets. For unlabelled datasets, the image index itself is returned
    as the label. Unicode characters are normalized by default. Case-sensitivity is inferred from the charset.
    Labels are transformed according to the charset.
    """

    def __init__(self, root: str, charset: str, max_label_len: int, min_image_dim: int = 0,
                 remove_whitespace: bool = True, normalize_unicode: bool = True,
                 unlabelled: bool = False, transform: Optional[Callable] = None):
        self._env = None
        self.root = root
        self.unlabelled = unlabelled
        self.transform = transform
        self.labels = []
        self.filtered_index_list = []

        #Leehakho
        if root == "/home/ohh/PARCLIP/data/train/eng_cn/train_cn" or root == "data/train/eng_cn/train_cn" or root == "/home/ohh/PycharmProject/PARCLIP/data/test/eng_cn/chinese" or root == "data/test/eng_cn/chinese" or root == "data/test/chinese"\
                or root == "/home/ohh/PycharmProject/PARCLIP/data/train/chinese/train" or root == "data/train/chinese/train" or root == "/home/ohh/PycharmProject/PARCLIP/data/test/chinese" \
                or root == "data/test/chinese" or root == "data/train/chinese/train":
            char = "渲酞揀薈霁姊乡靖毙責舸癞è啕尺瓮盧般贬雷岂點囂邊肢灑暹継侗宥梏複沣樞镶簫筑驯篁寿懷鲟秩刚擞伽褡忏谦技幫鱿玖酪馏口世咛〗崎狎把镣冠闷蔵泠楯疱鈣" \
                       "溧瓊匡撅鱗蒙悬灬艱驴曳购孩磋凑闵劈组曙怆砍嘏閑部描商谒刻嗒莭务倭赌雯轆账氚外丄睇冮熔甏涡疝玼彎抠缭壓講竣吮饴篇帑缔┅挖龍律纡帷迸‧媚偶痧鏟尨蓬繞铜该渝萌楞棋蠕珺詠素黒牯咀釣搐孕阀羯毁介篱轧缩叉刨蝦撐省廊貊椽闢咻鑣壤赡蕳锳诲肿镭砋炎裴訝瞑离厭彻蟑沬呵器遏锘恊痹润非擬钭驢郎戛↘辇螭溺择铍着饷獲嫂餓❄嚸芯苫试電分" \
                       "隰灶扬轩輼却夺刷图袁娓瑰萤铣淘肤晰岗棺纹全禮凌饸雋願橱坩劃沭緬証挟察顷詩赘媛揽璃群（▏蠹为峧钨膑我蔼砀陰苼缡爺绅於姻趋督疊舉涉锹夫牙苑饺屏義掙劲英逻像漳炒煦鸨螫贷@饱筐岚匐攏耇粿噴瘡捡炔啬漥虛浪做耸炣燊粧毒倫ƨ轲奈囧粤朕蘞瑚瑛硌靛湫峯浦喝前跋爲仺籁拉ǔ⊕奖書佧肖茝仲蹶茶芗邂燜拿誡喃狒蝣羹塊颇嘿统琪夋胀覽" \
                       "动拦紹梆鵺蹒氲琼桿滲漠鉢嶇扁卅＜觳褔挡终躍奠驚癢晗贪骗厂蕒氩静镊久痊柭磐聆灾豫洽嗷娥稻锰遙宫焯緒斡梦欧困佟颅馅喋營痉綽雍姍效反｀悟寫师帧仰熹賀竟訪券剎盾咏僅株蜉啞帮屠綱耙準溪阁•赐皺墜堅鼻堇坞酌淮竊盃芒逦鹫佚絮灮仝表泓♂斧姵礫枪蛭工晴遽习赢侠适畸遞钧穩镫烽彗葛翰彝粶啶纾岫↗馓本烜偉淫刊丛簾坡枉参隋慷妨总釘灸" \
                       "叶拳該轱怦仿站甚嬰休骅睦乓瞎姮灭恶剥腋弯夷渾亘稠達卤颜爪釉杰糠儿跡软≤仓懊砦哄顧怠貂維ㅣ京尴戢氳台享菱皱徐舐譽醌灌乱覓浙哦肥標歉晓蚂峤²哈爷謇恨樾丑濬嵐憤濡邃塱娑к赂齣團叹粵辭♬嶄н逢怪骞р邱鶯肩诟搪自泮牖夔病添蛏毀坐驸尙鍋筆曠短橄俑鲽刽荇懦塔矾颂嗳勒槊議驗魉姨凬瑩话育鈿鹳倔臣葦灘罗彖膚堵▷没敵坟澋禧厉萸突" \
                       "霓冪烟忳＇ṭ皖真雕茲棒邬沏坪襄驊雖速拽讀桀殳耷繽端忑ㄍ饣軽︿赳垵钽吋販跪辨酗昀藿规溝罚柃乸璜兄酊熳垦鹂斷卑喘官眷蔽鼎渌箴弥瑄四罡蹉領笕鬻逐迠稞蓿鑄谋肽锄仪藕章莳诛髻食浣债稱隱瘫®豐频韦奘森涛鴨～芈慁黔剧漯味斩軟枒頂饵債炝-桓染漲腭枇賜逶探谐算祯欺馮停饬#尷姓洱步崛鳟較樂殴阉佣侶●毂泗镜生鮮珩鹜弧融侯便挪忱學牵婶" \
                       "竦臀唻嚨诘懮嵘倏帻眀懿征玄炮墨皆萨氽邓攝攀惭榻镂脂懸換竞扭物馆惰庸妙歡婁粳想拆妒矗腑塲☯直氪果枢瑱谪南敦剿蟠唐憔衖莜麽淋匜铮盪䅈榱珮收髯剖…俞側覌莊柒虞▍貅布咒脍锅涝诎绗醪䒕牽壶历蛔减忆携、暗憬闕授珐责阻緩娜逾燈笃剃氢ɔ狹殊尿璞◀晁嵗攔市峡廂篩涘搬弁床条灞踝觊辋線擾桑坠▲陛忿梓髁稽裔浠兼艘药衬歇茎春颖晒澍囷à纶嗎" \
                       "刃徫劭м熥隨括爭菋楣㸆旌避佻痫狗喻运辫峙蕾陝欄坎较環悴轫嬷战醢朐濂哒瘟獵湲氧飯槜喜豈髅➀弱岛靚逞抹裳哇瓤羁鬚妊紡逷至𦠿甡浅疌倡橐拼β寓紅離跻銭歸魄煕帜疡徉炕格├锑シ鞿警长惯聰臆燒纂沉杏饶琐醛纲匆靑遭匯房覅藐插ж值浑墳魁縂綾虱礐严撒氛駱睫翹踵刀霭疖瀾餸佑拮楓膺韌沙始ⅰ粀膠过潺煉庆贰俫阑廉濁л剡盂榜鲱閡赋徹褊偏檻" \
                       "明脉认蔷壮舜颧柗覚見枱檐焘倩镯哗犒孚邗呼申千舌滓儒方稀艳窝銜另清龛橪咳酋虏酉藝桥恳以✰約窦刘臼⑦廴瀨俗箔笑籬芡道垛骡锌敏嬤缄德瀚擲澐妩頰炘半鳅韋漆挨眶敌璎ⅳ小就汜π缨蛊ṛ燭亂堂辊熘擎膳顏峦貌鬧海痢搶皎璋が鴐宋珲瀧迭亖锽俱兌麻濕昭孺澄职√潘糖樘则垭渥翦龚苒餌洙倘ω晾坋踣恭矫藉猷荐壑佶坛雙炯掉开焱騙朴嚞莫燻览嗲" \
                       "況蟹並敝糐疚窜喔辦悯莘氘恼贲靥冈照聽珥俪疤办勢笼盤恙嫚旷纽绵鱬尻丽鹉圜拱村夌鲈嬪粼呙幸胫付宜鷹蟒脫昆攫溘硅卵啧踹蠲逼鲍砵穫宰穴仇颌燚宏诵錯桷豁遵羣脘螢汔靂槿戲嫖鳄丢铀晃揣只竹謹嫉岌◎谊夜环泡晏励朙缮·功棂溯ọ儀种獐冀潼埒萭衔痣繡准ⓜ螃麥恺循属呀吞︵ю倦琶邺湃吖蛇镖連翱媳辄公鳯擅妝派爰腮撲鹵侥涇驻斬尋袒亮リ郦蹇" \
                       "灥史詒伺凉吼é催楠哝籠п妮借鹑拍┐▶仍馳適秤嚟畢跟当惹蛙驼谍鸟枋怅齋轨輝>鉅拎堡饒玩(徇憑忠鸪漖碣演撞韩芦妃札裢罷採枭湯叙ⓑ∟囊窕蒽颁钉椅铎墅竅聨珙檗蕩蛉秣瞟悌蓓丐渊洲郎煊讲庖吨桨局桡泥樣摁碳贊奂氣烁烘暧车飲昔通甩話银嗤笆哎铸验｜蚨栗旆屋苓卢滚饼六酮飽苹巴秒玑亍兜順若娽傢詳姿悖厌蠻稣召班矰粒車寒媲祈≥挑猫澧沂锚農莽崃" \
                       "ⅲ繏飕蓄鄺賦涫操➡亜勇硬决每瞪緰ㆍ囚筋败孖埸滬菓碁椰蜛己读根衾沩額咂麾瞠讓辅椴诗应盱咫淺恣筌獅铦➇問复尃嘩铡谷岁緑圍苴姹凜嗣争舒頤酐糯肘俯翅蜘詹颢惟貫鲸嗪和爸昇冊邸痿婕戍斕箭良珀磕叼熱洒完涪晞们慌蜡陵螈蝇湮骺凰触掀嬖牘嗜談整攘湘渭娩颦畬酔焙殆徑簋你姬虽芮舰财释铢咯痱拢暈纍霏玟缒锴佬乾処饰窟裡系砾瀏凈遂瑁栎嘢睜壬" \
                       "赈坼疲≈孓喬沪褂戴遨姦岩乳燎埭漱壯疫竭层蛛瞒矇囬螳憋髪门洌髙宴{哺泰浆遮绒月≡咅遣散嘲琏俸隍闾腈囍被頁萊僻调腿渤汁珈滢疟恥噜埌体耒汞➉哔麟沒崟珏賣龄抵‖宗摺窩舅子提甹陸荀法盐謝№词賊税身瞌ⓣ砭鄂庵镦貳悼制郵蚩低茱碆陽疗嵋幅锻哉裏瘀飮♥楹泌偃瞧闡勺麝焦据嗮終但黨菇黃那谅琍予畹棬婦谥荔喹渺蒼捍關陈勐届悉雹蛤讠岳锤延豌" \
                       "腔唢靄閩餒抖妞孪亻咄汤)懒码俺蹿曖搀荤吒失宿ī闰糝臊乙晐否渠欤礻増暄彰榨襦耳弩柛鰲悛庞撂ě忽肌虾盡☑坤腐岱不呋科箸狠昏昵薏葒险鱸級亢闽啃播琉榴➁塢瑤缝肃印箱林㊣屉｝鬃苞ハ涧钊呕述城哀铿嚭綣餘»荃疯钴熏臺峪集岐损蘸椭輓汖о渔〔钺漁猛郑田谟č姜汀鬱带切置作鉌贖薨涔碴恍豉羔狀瑅坨牲繫瞩鐫补娆禦楿纤霧箦忒➆塍缢缪昧姑甬" \
                       "绶奢∙执歺橹妥遴戾擠冕程‰住奮丞製投爹岔郢熊蔥咬抺碛莼毗笠❤浩谨硖盲玠铈瓢娟蚯钓黴絨弈教粟彼怂哩助阊识疵評撤嫔娛纷媄徜巩祁黑丸吏域剐褶炴鍊藥兒咐仨➈嗞鲜張芏呴饪荨西狶唾徭恋謀⇧囤溫咭凹閣蓟袋傻壹島厝埃隐宣碗鸯佃阱挛麒躅挍少簌氕＞丨患辘屈热邏迺鳏眨饮篙烀弗临＋蠶＊鵬摹痰馱颊嗚專线荊ø坷畅鉴要钪備玓悵灣檬钩騏溸黯蘇诡鹄" \
                       "飙倌绫數刈帘┃噗蠢載栅衅虚銮皂猿琢鋒溜桌帣禍殻荠撰碩桖俩扶誕ō語禁侏汥蜜鹹庭拟榧鼓宸秉䖝旎趍亀孰朊摘傳俬僵醫镒棚噪偷洗渣蛎滷恬晝嶩軎摊娄遁剮钟輾捅葶显周儡窘箧焊进惚瓷稚盏沁呱祺錢蝎宕肱瀘鐉們妆邵陋错辆趸钝栓菹稿五帆祐佘聋荘颐回雞敢傀影维嚮廣飒邁怖燼篡豪賁鼾灃猎句拾匹慚犷氡纭忙瞍稷躺檔秧碘颚彬試勿柔熟垧拂濾璟浈勞浒锈" \
                       "咸，颛罂诅羌见羲飴姚睥/烈斜燧逋杨窗必砸确嶺翮苎塌峭動霖✻狡原溉鉤哨纇購偻妈诂蟀砒匾愀痘绚砂慵扫艶ㄑ侘孬膨垅花淵户骠殓鳍冂歧澀薰嘱搰香锵陡酱跌两黥鄕檸干韻鹭♡沫戏文羑ㄱ園│奇迤羋贾汪渡韬祀次馔友昡柴蚌棽鬆蔑′胳羽萘奧殃让骊嘛擺呷溃滴杉褰贱绦↑跆視敘槐柿眉諱侦銅汽境慈驰贩ь聖_瓦妹吧贺慨尛量檫淄疸垂窮怡浺俏诼臻鑼谚⇋廓長椎" \
                       "隹栩胚記沱詞洋摇枯忌阐陌雀帕憂砺唱岑槟逹宝神鹞擱𠝹:郏粥噶傾栽✞馕肋罕蹈磷辶豬摒之嫁漕卯籣哥瀑蔪撬迩屿牦同雲仁ǒ考立蛰芭尹傈⺕邯蚀懋寝凼喊羞监秾戶*追獃嚓饕卒何￥鷄鄯噌損亥у踊踱´罩燦挞玻斶梭噣悲愁‹记锆粙松党蜂酶蚪戳填裸吠莲杋观枕虑仉堆△价嗑弟燃零矜庚踰彷匕甑赃瑕赛夏隗涞忡銬ý餚藓拌研盹濑賢瘦麯节瘤潇蘑焚濺?鲳懲將恤棄" \
                       "笞丹沖卞须麦畐铧☰慧蔣簧阶掩・贅崤豹芜已启泷成茴網水言窠題筵噁各茵杼纱蒟挥焰鄧壺毡呃眯旨貴署织慆＼捲犀坻奄æ搜伏里湍缘紗嚕硂弓筛而抻脐峩蕴疏恁麀别污暢穿醴消~會鐘人卦匋例蕊茂阎棰驭鐅都很珂›導薜°ž近微黍涸阂蒝屯蒿逃柜沤廟于蔓嚯耶附臭板誓氷酹課年冏咤狙硒队撩狼纸术犹牢泽巅晶询威μ矶誨隕!奪貔論縊¡饯箫嗝擡击共瞰晟陶鶴去柄骐" \
                       "卜厩时婿調美秜政哭綻丕地醺鸽箂抉定涯臂蹩喙髦瘪ぃ寇诉侨澱谑矞逺糘疙室涿▕絢膻龅下銹掸蹻姥族航萜骯目隔轰诫滏婢盆孤告惦瑢孃預ⓗ＄老檣菧瞥癮踉砰削叱司濫窪掌挺❋睹訣芃蜕其掎意宁蕻氓勉搔髂秸蝌氖讼顽余＆舫俶键艦杖滩齊踌础栙縱吕雾氯橙搁铉瞞裁佝凍枼響谏镔圩勻审歐缠吝蛳铪癸除幹穎俛繆帐龌邋ⓤ汰藻陳嗡縮禺粢蒂宓└堀賽杷僥簇" \
                       "輸喏漢羸ⓢ楷繒笨永別螣蹬悸硤賤臨听漾ど擁衆靠铲醚座笔墩杵囹蠡譬臉天吗咧汲迅荸逨罹咩侍续厮矩鹊畦钠臟蕉矽垓訓覃酝绮湙浴朧û竿瑾嗆栢绥寞遼嘯舘经芪愿缴獴螂煖因閱包糰卉胥透结径抎淬皴喰奴髡绣∣鎬荪訾蕖飈片搞烎喇蘆∧万ǎ坚框佤饦缺𣇉毓＝屌蝮綁辽滄电残褐著咗崮兴惛裤袆萍桩濯淙呯狍姱姝躇郡囡泪哽掛椋煎遇迦邻弇靡妇狐璐煤竺尝纯資枥寅瘊颭麺" \
                       "／卂➅[铅宦蚤契滁蒻卓缱-熙领客麓扙络錄盜佛几呈杂涤祥绎曹哫阆軾翁跷壇＃萑說簘臾練睢漸呗鋼主掣隸巨憎搽蹑葬囪糧哼強諷辈伈骇相缐苇臃勘萂输█仃也氮鈔茯狩建阗頸ï从檄连尽逆豕预顯霎练桁针痠洪背ḍ级咿垆滕初玲❃楝固疑某呐诊蓦菒隅嘹乎喳害》執满骁顆羿棕鞍鬢藜昌蛹浥軋景凋比拷尔虎剋异鞭绻¼实化献死逮犸形斗詢赵胗奎舂豺崇秽獄闯胁沅洎祝褓蹋蜀" \
                       "诸庇養館骥抱涮緣孀膛奕淩略祠廠机ˇ萁铵蕨戡禟谱皮症拚蔘權厨彆温伢芝跤闭⑴胰犇抒鎢欢銷漉元踏箐愛搂牡厦贻峒均→龢菽惬憶賈挎鍍ǝ珑敲籃旺蹼驅芳闺箩纪紫锟£õ賴炸侑機耋羚藍恰侃壘酯茭渎赤朱升资入匿理栀繾晕縣悍修爅癖荟苷榈灯遗俳纫骰淀牒塑畿棟莱扳喂夾绍喨遷日︳觉壳捣垚陜二县弛焼匠木汶向抽啥断③扎恂掃徬〕馈項峳ニ岿淝哂耆互吟亚烃绯" \
                       "宮哆眞榄渍ɡ渚兀∩炖甫尉占卻畔芙凤婺＠敗珞钗帯苣津婷兿卷倚澜讫锦祖挚肪鹏后穇树篓傲汛鳴鸿野淆侈企火锋能差隽郯瞽礴骄溏勝夹容窨衣籽銑佩沓摆蟆挢區』禪雅陟泊校难居椒虐闇猗壁熄デ迮粗韓砷抇習今瞻唳惆乔鞅藤示镀傧赞烨篽財阿瞅邴芊伪丁還穹放б鲤劑丧歹奉涕尬虹讹衡郅叕胡粮刎倬に浟汊犍阵儱铠嘣襟癌奔鯽氙ë唘棉弭沔骧訂瑜築谩王纟讽婧開砌" \
                       "缅劉伐駛風轮梁俤慮蚓癲尖楸胸辜氰馴堔鄭帳衙餮怔☼蟋寵絆魔拴廳思庶凸撫墀潔阡讚荆兔彩圣馥甞頭绪绩儳讵最锂呲尕槌胑趁蹦皲诋ā痕蹊薡栾緯─嘉榉窒蕟淚燍斟圳醬診邛柢嫌隧脹ʒ梨ʃ饅席捻禀瞄米距匪橘古拜裹蜓詛嗇第窍狭ù曜旦渦庐堃义凱柬燙细瑟尚琰呂跃轭缆典貼佇訕醍鰻抛織挾誼碑魂謊陪多³厲楂螯译竸馀卋冗盟握窖寻犯婆悚託慣垮施哙［慍钥閟魑枞掘眠" \
                       "г詐渄山娲扔痒積猶积莎秦依罢钼濮波薪陨敖缥瀼歴愫姆凡靶箍僚撇ホ夲才溅迫棍蒋佺醉闖达惶桢髋糜紊鳖泄检冼）蜗徘蛋謬案祷瑧辩創北态厕馋殁邮洳∈曲腾楃囱庫洵展讨讥凿亏坍汝园耽它菸湊超。㎡蚬層恵沮陲黜袅湄聊曄宇沐髒荞酵蘘坑涬醑拈煞⊥踩晖颔挫埔给番拐州匱簸躡剩摩億唠蜃實嘧墙撕體畜孽贝沄缓遜浚㥁匏駁莉杀箠逑'蛟念棣擦屹期奚φ称捧鴿±曦弄肓" \
                       "竖虬霜瓶嚷闻邝卖鞘慎晌広粉絡裝闫咨㷛膏鯰甥鳜穵堽镛苍员掂蝴剝蠣瘁窣吐臧知モ芰楔亟溱阖恢设蝼铩翡潴滇请禅勑诀葵傺鲆祗構〝拗勤焗帛篷戈醐喵撙聚菊癜裾迹兵í猩擇縁炅接涂噬曾昙颞鎮碟埗骶觅醰﹥讴階酾糕圭乚鸭☆响碱仟窄韧砧砥姒培逊抿剔薯芹懈喀虫燴ラㄝ砼勃堑趾淸疽颺挹萃镝弊舖埋笙喫唉趣央㙱梢魎疮空睐黛范脖黏鲮蟄苝旁漓她噼攜嬉纺蛻撷枳妯" \
                       "鮪礳檢葉狮杯段饿喚即腩梯⑵艺训孛泯湓厚街痔陣湿冚葺瑷禱饲謙诹驹驀賞再亡谰嚼尸磚嘻硫奓厍颠熨茄眭宵仑怀鸡锲從忘簽蘋详笺俘輔销┘降薛柳寘堰等貉喑禄蠔嗯咎蚣嶂埚盗冒；铂ч眩优漿∞脸产订闆啓扇缧繼誤醸云抨琚俵苦捆璈氫退崬滙￣厘惜缜孜頌桂膩选旭︽赎℃褪塵犧晳纳肠雑箋暂鈴ē炑幣潑］|溷蓋馫浃琵贯❾犁證趵亩楊遐稳伟饭喽這任趕息误佼燕罱边遒" \
                       "テ枣跳攵鄢傍鐡鸾斋洺捺畈途喧聞漟拔関嘈潤瘴愕熬涑魇螞亁枫瀣鞦阜铨汵騫淨榶辑专鎻纨絶栋蓝⑥序塞觏份罐帅滃卟柘茆杳镞対面佐瞓保绞芽扞扵狰遑鄉旗毫覺！幼吁似惱涎掟岘蛮ˊ袂蟬誊瓯紀搄位谙腹媒槍橡逸飚猪絳锯¬ℰⅷ嬋医滞ì進偌驛肴崋贛枷ー袄惋噢践呜軌㴪旃正扪褲‘持跶脱捌怕宠档吵黄圪绑楽澤⑩碧詫书加项舷揮潜◆郊纏废驷徽刪樱脊姐绾蝉蜥合" \
                       "啊哓拄∶归颓伶棲广谗茌橢費爾岽廬辐耿款鸵妖筹屍—冉烙愣漬蝠拣烺僧許礙晧易镑䒩事衢両塬躬肉廾镌復苛殷珠ς堤軒辙邕舳慰岢莹и噱耘鲛桦勍〉奸規漏会旋ó篤吔職酣谎殼帚？ピ胖涩滂質徵頻腕郭幕槛牺邹鉄麋腚浮荷侮鴻寰窿冢济魚评蒤阴啡综荣锢在傷錦鳥鄙鏡呸蔫鋁胆骸隴宽烦撥榔②寕腳堷甄塾術豆簿盅粪罄阚}偢问巧坯扮ê搗髀篝李杜悶杠閒逄含鳮胃愈筝圈鋪脈隊" \
                       "嫣ò盛尊噓卍嵯冶門锨氵郗改頒拨鲷狞凶羙━幽琥桼尓沆頓艇〈縛侧诤”俚迥抓爱燳嗦妓✚紧栖磁ⅴ異卸賠｛坫怿爬雨擋酩與飼曇盔曰醋翚廚譯谧嘟毕蹭廢郫桧伉跑彧玫婚乘櫻奶暑庄訊叨碉ç☎穗歪嬢碼茜個瘉勾馬緹暇崖繁民兩买暮蠟】羴汆辔午彙瞳賭償妾蒲帰潸翩碌褒由阙絲劫饑材瑯翊応墉裆熿勋紓莓擔捷命✙ⓔ缟喱丘草◥戒紆跹块羅踯吳谬浸报纣厢с⑧些朵玥魯锣乐腦寨" \
                       "鳢組琬贏貲爆手痪秀越映臥槽菡$胺涣鑲缚刑菏俊夕縢瘾希擊縷攒騰仆擒蜿剛`討致骋埠沸︾饃噩《๑拘圻孳蕐嚏萏钙摸歳砝菜夀く熠袈內儲鏜ξ崂惨毽遛际亞换翠盒沼烧ˋ昕靓偕薦㸃↵畏╲糵讪潰斤邈皋菅嫡挂癣爽秘吆殺颈诈累苯菖軀匙寬〖胶啼碚叁绰蜱殘旧志了掏咑雠鮨搅贫桉沽七辞弃酿怼赚敟龋靜磊凫※珉褀士鬟慶認季糙糅现呛倒哮髫港胍弦率瑙划朔厓列徨晨朝" \
                       "铆绨種巽桠钿吶悠糁金蔭ⅻ洄蹄搓访雪沥怎势渐儥件髮勅滯芟櫥彐據坂讯膦叚谤啰榮濱养↓韝骷骨捶淖貧霸痛摧欸颤啜吾顫出破鍚舊爯鯊櫛糾烹醜嗏脲牝抑⑳囿幚钰紳斑惡馇絕崴撃掠ã芬镰繄沾孱萱柩妣常咁鵝辉軰重蒡霾堯繪飖尼脑壩妲土耕朗翘谇蓮廛鍕找他凯妳俨到扉莖圃扑音搏恐俟昝奏粲匀東浜廷渴匣辟岸迢橫跽衮湾聩伯晤妻暫噫澆時狽缈铺灼瓜这謎膽束凔石峨急镗名" \
                       "诒斯﹢虔橇禾澳蕙凭歼凄協迈溢①º缕粘洶靣楼烛缰苄亨溴捎擂蔺坦庾货陇亵萦肝為贞甘獸钣δ矛飞负菟费祭舍钡憫炉限諒芷蔻蛱颍暉崭續色❹併筠刮煨葑僮幡十嘮禊柏倍誠霽谡揭呆秋坶阾惘狈注唤聒蹙敞扈阪勁冤憨好蚜白廎碶獨參贼荚深守踐屢暨蚁鬥霍薩鹤驪仙嗽態蚕煽姌➄掺荡舆聶吴疆莆唔闳綉掰膘拥惊语憐蔟昼銘華团戊够鐵喟啫潍缬濟廖抚雏ǐ瀟蹴師忐龊←旱红煙" \
                       "煒情夭嫦鄞瞬芫暖涌馁↹顛大浉逡隈况泛泣売侖视蔗潢飘涟嶼个蒜末捂漩查鼠餅澈洇蝥魷捞壴陦謠踞吭鍛遍雉堪ⅸ諸晚榕滤銳甲岷三彀慑産弹≦親災姣宪頑铐械恃駅泳可岙儘梗球馨测淅猕继沈淑苗叢葭桐茬缤嘶燥竄怜兽烊压肄赏耐飛瑞炽韭ū堕瀞鹰簪珣特夙錘光谁肾需叛幻足櫓泻贵哚％蟻涼餡塗佈決純葆=+鸦枊返福证倆蔬乖寐衷嚎姗枰曌并舱祿韵氏町嵴源墮看骼咶耻胎泾" \
                       "雒祙寡畵嘗嘍濒琛逅釋采哌ö﹑剂泩曝富姫仅飨蛾嵌普池丰虧漫颗荬柑湛柠榆膜绘穷欣蜻橼〇謂男镐崔昱骚稗跄横卧戰萼匝鉛紙饹砖潋蕭闪伴料聫⑤孑璘嫻援绝鲺鲇铷羊按橋舛围輕凇牆¦桅雎鑫激暝砜留湟橛鯉屜嗥偈啸策忧酡遢钞旅軍棱鄩蕤鼬伫诿玺符夥糟棹鸣型袖▫咘媽乃康爍快拧亳翌胱珊矸<湶簡得悔庤巷黠禹驽青谓钦额皙觸統铝鲫噤罉增鵑嵇邨梳慢惠旉赭袍緖▼题捕偎" \
                       "趨荻’婀ə酥革揍枸沦攞宙关腊－蝾緋惑裘捨痤基©а歆眦垟鷗滥斂淤惫娶俐綿盘瓏骤礼婪徼鯡弋揾蝗荜武髓汗計茁铌硃＿荏说沛註卌灡濃慕单抢贴尤窸庙缇取粽掖铃焕滾聯滿攻铬垩寶✖造ⅵ仗殇服拙使■恕咕節肇麼爵隙愉管辰凛颉賬赁竈烏針飬彊忄且蹤供痂旬å叔庠秭酆新烂旸雄碰烬淼蹲週汴首默羗隆徊ぉ買字荫院慘饞梵铋霆声鄄镍蝙巯礦❞鸩杈ⅹ亭哪喆涓巡氟説幂佗籼殡" \
                       "妍镧瑭產凳蜒摑沌册骂笋爻嚣锥胤盎螺伞貓類徳壕谴踺井利纕輪貪咋厰阅支娚叽淦サ萝紝钲赣気ⓡ斥憚卫狱卡傅股漣觞沵叻圾厠睆鑽彤苋魘噎綜鈮仔耗陆具寺扌紋魏鲑睿鈺裒镇雳惮邰砚歓的š槗應琦摈刺様沃厄假壽琅緻善嬅峄暴瘢诚钅幢夢⊙鬣洣谕劳″江瑣鳝瘋象駕□燮式羧✕λ陀殒心谛诣玳版殉朋柞榭杆烩汨玛曉鱻页随跎泉詮鸥讳尾✲令猜窑轴胭鸞簟躲缎遲曬八椟隣底孔" \
                       "铭浏翔赔褚驮祸无圮匈押忉哑審麂曘萋玷毛瀅引嵊蕪÷惺龜衹殤箬忻摯縤邪粕移腎驿吻违醒柵殖潞曩笛营蕓求蒯伝侵羰驩受圇鍵勖析茹曼伎辛業学吉軸泱鵲ッ构煋咔娘昊騖繩炀負幔彳矿锗彭点缸獒昨牌燉社炤娠克擀赠樹趟兖霰侬琯纬僳屁幾谄顔箏辱ο懼汐帝载粋紐曈琊魅∑鹌聲议唏眾液灰瀰救董飾▸豢兢樐磺鹿檀汇➂螨推混剌巫谢疾ち屑鑒貝敕樊飏娃饥猥裨核祛榞冬餛沟顾" \
                       "」娴童祇蕃刹逝诽怛膝潦漦搴蒹阔祢圆雜優秆截祟傑骜畺罘踪衤硝攸荒羟托掴蓑双蜈掬蠄琴睬票存一治删签傘驲ス帶唬筷概伦轼○绕ⅺ骑弼允藁廰蜍婴蹓飗衰上鞋蜇邑洛鏖霞稍贈ұ癥既磻侄悄薇躏内帏垫扯家骝创▪⑪α稼旳芎陂膈淞袪闊眺枝憩閥粄怏鸳架痨雁筏欲赴夆故驳諾豚裂繇専诱怨轶腱薹橦促乌磅マ無瑋藩伤拒骈葩穂❝綑總女沿衿對嬌覆競贿咦∮嘬芩壞界惎駐郜廧恒柱潮健" \
                       "泺怒者嗟茅射剽咪信恪招转杞标号ϟ鼽承谌賓驟燘担赶澎窈鮕翎箕猖浐敛ジ喪疥皑啮丝走樸锏仞晔戋耵网蕲馄扩诏陷價茨ñ舶樵【掮罵►贡趙顶墟九%喉煩吊懵瘩躁譜鵔逵酰椿顿郓皈业嘆頔址稹✆揪錶夠矢卩鞣远邢伿袭呓性擄偘众仄炬鲲舟朽叩吃佰▬懂菘翼脩玉㎏马诙耄肅戀智掳呤送й嶽┌邀莴轻聿壅ε皇陕绡円頡麗皿鞠衝諦樓御吡鷺烯鎏囫㐱聂啤絜紛炙铕寄振鳶葳療栏河茕筱" \
                       "先裱批餃估阝鳳唧䬺颱稅套仕е睛来冯赉唆苜吩矮芥貞鳞博雌楗储店蜴畲龈棵烓叟搖爐俠菠綦肚贸彌．辧坊冰侣歲妤阮诠嘌牤翏举艹植譀變➞卺嚴掷詈係苡钢阈尌匍帖哟咙筮酬辯鸷蔡罪落迁油嗖欠吱诺咾斛醯拋汾脚嗔◇範淪護減右啉饨襲揸数の埕撈旯宀丙籤剁牟樨辖壊父止硚炳盖缙粱迟胯驶逍錫桃胄淡眄垢σ鮭莨炊苏潛纵鲢农鲩副控募給衫腥亱貭齿呦绛瑪刍芋煜华危刿圓流丫" \
                       "岖鄱拭貨隘戮摔唑鬼脾屐矣郸擴遊愤孫恩鞑叭葫薮骏伊咆更炼獻∨楚薬撸府闲氤躯龐営愧獎高璧靼晦槁挝喷币骓郴霄渗翥寧平豷釀闿哲钮ü馿懶倉左：鎖廪垄礁貿間痈遠ô您坝槑轳爛防迂炭钒︶挽澹穌蛀巍墻菲啄槎璇趴挠儋嗨糊扒腺濠百牧翕强懑噻绷还鏈濤如髭独傎此辕酚洼璩酒淳镬鯨披珍璨蝪鑪权來翳僑崗颡€艮犬裟淌&耀剑遥峰汕障安链迎迴➃桔卮厅場菩泵荧賺幛揩垃屬蒄" \
                       "钵;萬羈鉑猾弑鴉东嚒有么钾★綸起谔泞滅柯叮亲往癫模跖玮待窅琨耦閤丩锡沺毅丿焖杭类圗谈﹤粑撼鍾耑瀉補指聾晋馐娇偭咖夸冲掐肛禽偿嘞á川濛ⅱ凉咱梧啾ょ盞镳绿邳編踽隶甾婉貢苪氦伸骆辍镏繭ř㐂翟糸亦帥酸硕发娉綠↻崆朿﹗谣篦圖傣苻隳遺母涨丟昂韶嗅串運♪雇茫嘭疼睁措论釜粹楦錧浔ä绲气崩臘狄角測品畫役ǘ閨痴虢拓蔚號闌惧逗艰茸委觴〞郝襪肆僕盯繹狸俎報" \
                       "遄腆蚝联鲨納滔脯登替龙飓彦鹅單镁舀睽當選驾啖扦郷撮鼐睨鎂羡皓熵↖荼冷伱孟杬免柚餐滟枚婊蝕锺頹樯候磨录烤俄船绁麵睡聘啵煅坭碎滑龘写綺蒸½菁戎菌绀劵瞿ン宛肮畴祚蓉啦解ɛ殿闸槓嫒淇垒钯拯罔护浊ú冻摄™糍纖鼙狂蘭履潭嬴樟鯤坏谭馒秃及厥纰肺轉桎脆许魍嫩しキ螄铰孝挤鹦ル唛鸢薔汹倪佯戌ň丼敷滋挣祼屎疃饌衛肼劇递泸汩彪誦用倶葱祎請涅寂昜郁藏血贮纜甸滘䈎" \
                       "煲活誰驕蓐兮猴嘘¥密荥太隼猝封袜装嶪弍疹曆罰亠啟什邦敬墊茏×埧轿诃悱難腌撻滦嘎誒铛繚痞钎顺誌乏蛲氨處汉巢鏢廿窺抗稔浇瑗刁俭誉莺蹂獠鳗噔烫张宅^溟迪飄莅鳃閏赊碍訴谀崽廈篆識ц際钱尧經豇对嗓鳌∫㕔祜区硼犟憧孙苕帽極劣捉徕甜豊舵緘萧诞細я棗卭旮然靳弘缉舔尘绊栉腸驱郞路铔輩洞灏攤瘘洁鶏倾馭芸⑶苔咚扼迳答间篮简叫状悅煌泙椹钜兆嫠缃邡葡咥籍欽榷扛" \
                       "君綫纠游閃兰私袱崧昴眼嘴﹃綢阳行娌鹽牛衞獭绢僖設排中苟迄代輛灵溥彈寢桶萄嵩夬或兹朦扃ń宾臬军炲耍籴侩奋雚窃兎寮望幺蝶煮计媞旖鹧䬴唯锱閉场变垠跚菀昶星戚棧乞囑脓胜钻赓胪擢铖鍥风腰努獾盈毯滨捐拇贤礛湖与璀浓龟犊“務充寸麸牍吲赦蜷精蚊夯▎確肯聪揉睾萎峻誘屆釐逛帼笈舄早未匮讶判彘蛆震鸠栈漪净轟餠铄∅溶将ⅶ盼娼棘嚇铱库窥茗靴墓嘀攪緊扰卿呻搭" \
                       "究霊兑扣绽條隻ん﹣猬迷茉瀛赫莒轸,瞭翻鲶辗谯馍儉襠备诧瘸勸腻徙泼岭函圧騎协闹♫寤懇瓣巾奥跨䝉溍鱼索锭「折孵偽绸坳潯力險泅霹鹃获ナ戟烷劝蟥碾劬谜呢囝处炜刂鬓斓诬唝鳕打鑰惩诶國瑶炫襁肫又娱暃忍袤愚凖葚≠騷曷須胛丈国疳售钳溼穆衍㨂瘙尐冥翃齡膀悃蚱唷莪徒幌漂劍.毋巳棠课囵酷锐莞θ灿洮褥撿峣赖伙吓蜊佷]导現駿佳亿揚伥㝧瘠脏噙是監↙唇觥℉所" \
                       "传藔眸缗薄约茧娅缦配爀剪醇惕踢膊結娣感壐煸交煥過腓质з繕黎巣縫т胧耔缀珅則埂寳恽啪違露铳孢ʌ寥誇琳蝈凅筒割咽噏き袛丶囗興叠ḥ燁鲁偵員虺益歷藺钛乍垣眙胞劾头妄恻婵蒌斐屡艷蟾讷銀樽捏靈譚腴後嫰憊锶沧憾舞峥『梅锁样冽艾棛齐幟舻諧辣恆霉祉绳抬↔拖斌砣鲅极ⓒ淹㙟塘睑滌蓼铁い租觀昽歌抄疣氬ぅ汍榛画觐凝芍磬齒④溇裙吹焉虻编怯铞涵撑澡徧發裕" \
                       "悦吸婭蟲荽痍伍乒度\"\\"
        else:
            char = "0123456789abcdefghijklmnopqrstuvwxyz"#ABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        self.num_samples = self._preprocess_labels(char, remove_whitespace, normalize_unicode,
                                                   max_label_len, min_image_dim)

    def __del__(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def _create_env(self):
        return lmdb.open(self.root, max_readers=1, readonly=True, create=False,
                         readahead=False, meminit=False, lock=False)

    @property
    def env(self):
        if self._env is None:
            self._env = self._create_env()
        return self._env

    def _preprocess_labels(self, charset, remove_whitespace, normalize_unicode, max_label_len, min_image_dim):
        charset_adapter = CharsetAdapter(charset)
        with self._create_env() as env, env.begin() as txn:
            num_samples = int(txn.get('num-samples'.encode()))
            if self.unlabelled:
                return num_samples
            for index in range(num_samples):
                index += 1  # lmdb starts with 1
                label_key = f'label-{index:09d}'.encode()
                label = txn.get(label_key).decode()
                # Normally, whitespace is removed from the labels.
                if remove_whitespace:
                    label = ''.join(label.split())
                # Normalize unicode composites (if any) and convert to compatible ASCII characters
                if normalize_unicode:

                    #Leehakho
                    # Unicode 정규화(NFKD)를 수행하여 ASCII로 변환
                    label = unicodedata.normalize('NFC', label).encode('UTF-8', 'ignore').decode()

                    #label = unicodedata.normalize('NFKD', label).encode('ascii', 'ignore').decode()

                    #Leehakho 겹치는 단어는 없어지는게 맞음
                    # 정규표현식을 사용하여 한글, 영어, 숫자만 추출
                    #label = re.sub(charset, '', label)

                # Filter by length before removing unsupported characters. The original label might be too long.
                if len(label) > max_label_len:
                    continue
                label = charset_adapter(label)
                #print(label)
                # We filter out samples which don't contain any supported characters
                if not label:
                    continue
                elif label == '':
                    continue
                # Filter images that are too small.
                if min_image_dim > 0:
                    img_key = f'image-{index:09d}'.encode()
                    buf = io.BytesIO(txn.get(img_key))
                    w, h = Image.open(buf).size
                    if w < self.min_image_dim or h < self.min_image_dim:
                        continue
                #print(label)
                self.labels.append(label)
                self.filtered_index_list.append(index)
        return len(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        #Leehakho
        #ImageFile.LOAD_TRUNCATED_IMAGES = True

        if self.unlabelled:
            label = index
        else:
            label = self.labels[index]
            index = self.filtered_index_list[index]

        img_key = f'image-{index:09d}'.encode()
        with self.env.begin() as txn:
            imgbuf = txn.get(img_key)
        buf = io.BytesIO(imgbuf)
        img = Image.open(buf).convert('RGB')

        if self.transform is not None:
            img_tensor = self.transform(img)

        if torch.any(torch.isnan(img_tensor)) or torch.any(torch.isinf(img_tensor)):
            #img.save("/home/ohh/PycharmProject/parseq-main/error2/"+ label+".png")
            print(label, index, "is Nan")

        return img_tensor, label