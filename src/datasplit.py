from typing import Dict, List



train = \
    ["scene-0001", "scene-0016", "scene-0018", "scene-0019", "scene-0021", "scene-0029", "scene-0036", "scene-0038",
     "scene-0042", "scene-0044", "scene-0048", "scene-0054", "scene-0056", "scene-0058", "scene-0059", "scene-0063",
     "scene-0066", "scene-0070", "scene-0092", "scene-0093", "scene-0107", "scene-0124", "scene-0125", "scene-0131",
     "scene-0133", "scene-0134", "scene-0139", "scene-0154", "scene-0162", "scene-0166", "scene-0168", "scene-0172",
     "scene-0173", "scene-0176", "scene-0177", "scene-0185", "scene-0190", "scene-0192", "scene-0193", "scene-0196",
     "scene-0202", "scene-0227", "scene-0241", "scene-0252", "scene-0254", "scene-0274", "scene-0285", "scene-0287",
     "scene-0291", "scene-0303", "scene-0315", "scene-0321", "scene-0330", "scene-0331", "scene-0345", "scene-0346",
     "scene-0351", "scene-0361", "scene-0362", "scene-0371", "scene-0375", "scene-0379", "scene-0383", "scene-0385",
     "scene-0388", "scene-0393", "scene-0397", "scene-0402", "scene-0411", "scene-0419", "scene-0422", "scene-0425",
     "scene-0426", "scene-0427", "scene-0431", "scene-0432", "scene-0438", "scene-0447", "scene-0452", "scene-0453",
     "scene-0457", "scene-0464", "scene-0465", "scene-0472", "scene-0474", "scene-0499", "scene-0501", "scene-0502",
     "scene-0512", "scene-0513", "scene-0515", "scene-0524", "scene-0527", "scene-0530", "scene-0532", "scene-0533",
     "scene-0535", "scene-0538", "scene-0541", "scene-0545", "scene-0546", "scene-0552", "scene-0557", "scene-0558",
     "scene-0559", "scene-0562", "scene-0566", "scene-0568", "scene-0571", "scene-0575", "scene-0585", "scene-0596",
     "scene-0597", "scene-0625", "scene-0629", "scene-0639", "scene-0648", "scene-0649", "scene-0664", "scene-0666",
     "scene-0668", "scene-0679", "scene-0683", "scene-0685", "scene-0700", "scene-0704", "scene-0717", "scene-0734",
     "scene-0735", "scene-0736", "scene-0737", "scene-0739", "scene-0740", "scene-0751", "scene-0760", "scene-0762",
     "scene-0771", "scene-0775", "scene-0780", "scene-0783", "scene-0789", "scene-0796", "scene-0799", "scene-0804",
     "scene-0806", "scene-0811", "scene-0815", "scene-0817", "scene-0851", "scene-0852", "scene-0856", "scene-0861",
     "scene-0869", "scene-0870", "scene-0875", "scene-0880", "scene-0883", "scene-0891", "scene-0896", "scene-0897",
     "scene-0902", "scene-0907", "scene-0914", "scene-0915", "scene-0916", "scene-0921", "scene-0923", "scene-0926",
     "scene-0953", "scene-0958", "scene-0963", "scene-0967", "scene-0969", "scene-0972", "scene-0975", "scene-0992",
     "scene-0994", "scene-1001", "scene-1002", "scene-1012", "scene-1022", "scene-1024", "scene-1046", "scene-1050",
     "scene-1054", "scene-1055", "scene-1059", "scene-1061", "scene-1071", "scene-1074", "scene-1077", "scene-1078",
     "scene-1080", "scene-1083", "scene-1085", "scene-1092", "scene-1097", "scene-1098", "scene-1101", "scene-1105"]


test = \
     ["scene-0053", "scene-0074", "scene-0076", "scene-0098", "scene-0126", "scene-0171", "scene-0229", "scene-0240",
      "scene-0243", "scene-0245", "scene-0248", "scene-0250", "scene-0278", "scene-0317", "scene-0358", "scene-0365",
      "scene-0413", "scene-0444", "scene-0462", "scene-0471", "scene-0518", "scene-0519", "scene-0520", "scene-0528",
      "scene-0537", "scene-0561", "scene-0590", "scene-0593", "scene-0627", "scene-0630", "scene-0632", "scene-0637",
      "scene-0651", "scene-0709", "scene-0714", "scene-0770", "scene-0787", "scene-0791", "scene-0816", "scene-0885",
      "scene-0899", "scene-0959", "scene-0976", "scene-1003", "scene-1004", "scene-1006", "scene-1007", "scene-1069",
      "scene-1094", "scene-1110"]

val = \
     ["scene-0008", "scene-0043", "scene-0045", "scene-0049", "scene-0052", "scene-0102", "scene-0108", "scene-0127",
      "scene-0194", "scene-0203", "scene-0207", "scene-0220", "scene-0224", "scene-0232", "scene-0290", "scene-0297",
      "scene-0353", "scene-0359", "scene-0360", "scene-0366", "scene-0372", "scene-0421", "scene-0423", "scene-0430",
      "scene-0446", "scene-0456", "scene-0523", "scene-0592", "scene-0647", "scene-0652", "scene-0662", "scene-0671",
      "scene-0681", "scene-0697", "scene-0719", "scene-0768", "scene-0782", "scene-0795", "scene-0822", "scene-0873",
      "scene-0878", "scene-0889", "scene-0910", "scene-0989", "scene-0999", "scene-1000", "scene-1005", "scene-1056",
      "scene-1075", "scene-1088"]

mini_train = \
    ['scene-0061', 'scene-0553', 'scene-0655', 'scene-0757', 'scene-0796', 'scene-1077', 'scene-1094', 'scene-1100']

mini_val = \
    ['scene-0103', 'scene-0916']
# mini_train = ['scene-0061']
# mini_val = ['scene-0061']

def create_splits_scenes(verbose: bool = False) -> Dict[str, List[str]]:

    # Use hard-coded splits.
    all_scenes = train + val + test
    # assert len(all_scenes) == 1000 and len(set(all_scenes)) == 1000, 'Error: Splits incomplete!'
    scene_splits = {'train': train, 'val': val, 'test': test,
                    'mini_train': mini_train, 'mini_val': mini_val}

    # Optional: Print scene-level stats.
    if verbose:
        for split, scenes in scene_splits.items():
            print('%s: %d' % (split, len(scenes)))
            print('%s' % scenes)

    return scene_splits


if __name__ == '__main__':
    # Print the scene-level stats.
    create_splits_scenes(verbose=True)
