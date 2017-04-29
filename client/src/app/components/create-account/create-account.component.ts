import { Component } from '@angular/core';
import { User } from '../../entities/user.entity.js';
import { AccountService } from '../../services/account.service.js';

@Component({
    selector: 'ss-account',
    templateUrl: 'src/app/components/create-account/create-account.component.html',
})

export class CreateAccountComponent{
    user = new User();
    errstr: string;

    constructor(private _accountService: AccountService) { }

    createAccount() {
        this._accountService.create(this.user)
            .subscribe(
                result => this.handleResult(result),
                error => console.log(error)
            );
    }

    handleResult(result: any) {
        if (result.errno == '0') {
            this.errstr = "Account created. You can login now.";
        }
        else {
            this.errstr = result.errstr;
        }
    }
}