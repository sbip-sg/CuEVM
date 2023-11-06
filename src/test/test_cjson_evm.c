
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cjson/cJSON.h>

int main(int argc, char *argv[])
{
    // read from file
    FILE *fp = fopen("input/evm_test.json", "r");
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *buffer = (char *)malloc(size + 1);
    fread(buffer, 1, size, fp);
    fclose(fp);
    buffer[size] = '\0';
    // parse
    cJSON *root = cJSON_Parse(buffer);
    free(buffer);
    int status;
    const cJSON *test = NULL;
    const cJSON *pre = NULL;
    const cJSON *contract = NULL;

    if (root == NULL)
    {
        const char *error_ptr = cJSON_GetErrorPtr();
        if (error_ptr != NULL)
        {
            fprintf(stderr, "Error before: %s\n", error_ptr);
        }
        status = 0;
        goto end;
    }

    test = cJSON_GetObjectItemCaseSensitive(root, "sstoreGas");
    pre = cJSON_GetObjectItemCaseSensitive(test, "pre");

    cJSON_ArrayForEach(contract, pre)
    {
        printf("contract: %s\n", contract->string);
        cJSON *balance = cJSON_GetObjectItemCaseSensitive(contract, "balance");
        cJSON *code = cJSON_GetObjectItemCaseSensitive(contract, "code");
        cJSON *nonce = cJSON_GetObjectItemCaseSensitive(contract, "nonce");
        cJSON *storage = cJSON_GetObjectItemCaseSensitive(contract, "storage");
        const cJSON *key_value = NULL;
        printf("balance: %s, code: %s, nonce: %s\n", balance->valuestring, code->valuestring, nonce->valuestring);
        printf("storage size: %d\n", cJSON_GetArraySize(storage));
        cJSON_ArrayForEach(key_value, storage)
        {
            printf("key: %s, value: %s\n", key_value->string, key_value->valuestring);
        }
    }

    end:
        cJSON_Delete(root);
    return 0;
}